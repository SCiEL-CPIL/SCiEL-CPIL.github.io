"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Global Bayesian Linear Regression with Thompson Sampling.
    
    Instead of maintaining separate models per arm, we augment the context with 
    one-hot arm indicators to form a single linear model: 
    reward = w^T [context; one_hot_arm]
    
    This allows the model to learn a global linear structure and share 
    statistical strength across arms via the regularization prior. 
    We use a time-decaying covariance (exponential forgetting) to handle 
    non-stationarity. The posterior is maintained in the information form 
    (A, b) for stability, but we solve for theta and sample efficiently.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.total_dim = dim + n_arms  # context + one-hot arm
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # Safety margin
        
        # Forgetting factor: beta in [0, 1]. 
        # beta=1 is no forgetting (stationary). 
        # beta<1 applies exponential decay to old information.
        # For non-stationary, we want to forget faster. 
        # Typical values: 0.99 to 0.95. Let's try 0.98.
        self.forgetting = 0.98
        
        # Initialize posterior:
        # Prior: w ~ N(0, sigma^2 I)
        # Information form: A = (1/sigma^2) I, b = 0
        # Let's set sigma^2 = 1.0 for simplicity, so A = I, b = 0.
        # Actually, to be more flexible, we can use a stronger prior if needed, 
        # but I is standard.
        self.A = np.eye(self.total_dim)
        self.b = np.zeros(self.total_dim)
        
        # Precompute identity for speed
        self.I = np.eye(self.total_dim)

    def _augment_context(self, context, arm):
        """
        Creates the augmented feature vector: [context, 0...1...0]
        """
        x = np.zeros(self.total_dim)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            # Fallback: return the arm with highest prior mean (0) or random
            # Since prior mean is 0, all are equal. Return random.
            return int(np.random.randint(self.n))
            
        context = np.asarray(context, dtype=float)
        
        # Compute posterior mean and covariance
        # theta_hat = A^{-1} b
        # Cov = A^{-1}
        
        try:
            # Solve for theta_hat
            theta_hat = np.linalg.solve(self.A, self.b)
            
            # For Thompson Sampling, we need to sample from N(theta_hat, A^{-1})
            # We can compute the Cholesky decomposition of A^{-1} or A.
            # It is more stable to Cholesky factor A and then invert.
            # A = L L^T. Then A^{-1} = L^{-T} L^{-1}.
            # Sample z ~ N(0, I).
            # theta_sample = theta_hat + A^{-1/2} z
            # Let v = L^{-T} z. Then A^{-1} z = L^{-1} v.
            # Actually, standard trick:
            # 1. z ~ N(0, I)
            # 2. Solve L^T y = z  => y = L^{-T} z
            # 3. Solve L delta = y => delta = L^{-1} y = A^{-1} z
            # This gives delta ~ N(0, A^{-1})? 
            # Wait. If A = L L^T, then A^{-1} = L^{-T} L^{-1}.
            # Covariance of delta should be A^{-1}.
            # Var(L^{-1} L^{-T} z) = L^{-1} L^{-T} I L^{-1 T} L^{-1} ... this is getting confusing.
            
            # Let's stick to the standard robust method:
            # 1. Compute Cholesky of A: A = L L^T
            # 2. Generate z ~ N(0, I)
            # 3. Solve L^T y = z
            # 4. Solve L x = y
            # 5. theta_sample = theta_hat + x
            # This x is such that x ~ N(0, A^{-1})?
            # Let's check: If A is identity, L=I. y=z, x=z. x~N(0,I). Correct.
            # If A = 2I, L = sqrt(2)I. y = z/sqrt(2). x = y/sqrt(2) = z/2.
            # Var(x) = Var(z)/4 = 1/4. A^{-1} = 1/2 I. 
            # Wait, Var(x) should be A^{-1} = 0.5 I. But we got 0.25 I.
            
            # Correction:
            # We want x ~ N(0, A^{-1}).
            # Let A = L L^T.
            # We want Cov(x) = L^{-T} L^{-1}.
            # If we set x = L^{-T} z, then Cov(x) = L^{-T} I L^{-1} = A^{-1}.
            # So we need to solve L^T x = z.
            # That's it. Just one solve.
            
            L = np.linalg.cholesky(self.A)
            z = np.random.randn(self.total_dim)
            # Solve L^T x = z
            x = np.linalg.solve(L.T, z)
            
            theta_sample = theta_hat + x
            
        except np.linalg.LinAlgError:
            # Fallback if singular
            try:
                A_jittered = self.A + 1e-6 * self.I
                theta_hat = np.linalg.solve(A_jittered, self.b)
                L = np.linalg.cholesky(A_jittered)
                z = np.random.randn(self.total_dim)
                x = np.linalg.solve(L.T, z)
                theta_sample = theta_hat + x
            except:
                # Total failure, return random
                return int(np.random.randint(self.n))

        # Evaluate all arms
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Augment context for each arm
            # Optimization: We can construct the augmented vectors efficiently
            # x_a = [context, e_a]
            # score_a = theta_sample[0:d] @ context + theta_sample[d+a]
            
            # This is O(d) per arm, total O(n*d)
            base_score = theta_sample[:self.d] @ context
            arm_scores = theta_sample[self.d:]
            scores = base_score + arm_scores
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # 1. Apply forgetting
        # Decay the information matrix and vector
        # This effectively reduces the weight of old observations
        self.A *= self.forgetting
        self.b *= self.forgetting
        
        # To maintain SPD and prevent drift to singular, we re-add a small prior 
        # or just rely on the new observation. 
        # However, repeated multiplication can cause underflow or loss of 
        # definiteness. A common trick is to add a small identity term 
        # proportional to the decay, or just reset periodically.
        # Better: Add the prior back in proportion to the decay?
        # If we view this as a sliding window with exponential weights, 
        # the effective prior is sum_{k=0}^inf beta^k I = 1/(1-beta) I.
        # But our A starts as I. 
        # Let's just add a small regularization to keep it well-conditioned.
        # Or, simply add the new observation. The forgetting handles the rest.
        # To prevent A from becoming singular if beta is close to 1 and data is poor,
        # we can add a small epsilon * I every step.
        self.A += 1e-6 * self.I

        # 2. Augment context
        x = self._augment_context(context, arm)
        
        # 3. Update posterior
        # A <- A + x x^T
        # b <- b + reward * x
        self.A += np.outer(x, x)
        self.b += reward * x
        
        self.t += 1
# EVOLVE-BLOCK-END
