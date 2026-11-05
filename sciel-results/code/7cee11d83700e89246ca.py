"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a Bayesian-like Thompson Sampling
    approach with adaptive forgetting. This is a structural change from the deterministic
    UCB approach. We maintain a posterior distribution over the weight vector for each arm
    (approximated by a Gaussian) and sample from it to decide actions. The forgetting factor
    is adapted based on recent reward variance to handle non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        # Prior: A = I * lam, b = 0 => theta ~ N(0, I/lam)
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting: start with a strong memory, adapt if rewards change
        self.beta = 0.95
        self.min_beta = 0.80
        self.max_beta = 0.99
        
        # Track recent rewards to detect non-stationarity
        self.recent_rewards = []
        self.reward_window = 50
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin

        # Random seed for reproducibility in sampling
        self.rng = np.random.default_rng(42)

    def _adapt_forgetting(self, reward):
        """
        Adapt the forgetting factor based on recent reward variance.
        High variance in recent rewards suggests non-stationarity, so we decrease beta.
        """
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= 10:
            # Use coefficient of variation or just standard deviation
            # If rewards are changing rapidly, std is high.
            # We normalize std to adjust beta.
            std = np.std(self.recent_rewards)
            # Map std to a beta adjustment.
            # If std is high, we want lower beta (faster forgetting).
            # Let's say if std > 1.0, we push beta down towards min_beta.
            # If std < 0.5, we push beta up towards max_beta.
            # This is a heuristic.
            if std > 1.0:
                target_beta = self.min_beta
            elif std < 0.5:
                target_beta = self.max_beta
            else:
                # Linear interpolation
                target_beta = self.max_beta - (std - 0.5) / 0.5 * (self.max_beta - self.min_beta)
            
            # Smoothly transition beta
            self.beta = 0.9 * self.beta + 0.1 * target_beta
            # Clamp
            self.beta = np.clip(self.beta, self.min_beta, self.max_beta)

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback: pick the arm with the highest mean estimate
            best_arm = 0
            best_mean = -np.inf
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    mean = theta @ context
                    if mean > best_mean:
                        best_mean = mean
                        best_arm = a
                except:
                    pass
            return best_arm

        context = np.array(context, dtype=float)
        
        # Thompson Sampling: sample a theta for each arm from its posterior
        # Posterior: theta | data ~ N(theta_hat, (A^{-1}) * sigma^2)
        # We assume sigma^2 = 1 for simplicity, or we can estimate it.
        # For LinTS, we usually sample from N(theta_hat, A^{-1}).
        
        scores = []
        for a in range(self.n):
            try:
                # Compute theta_hat = A^{-1} b
                theta_hat = np.linalg.solve(self.A[a], self.b[a])
                
                # Compute covariance A^{-1}
                # For efficiency, we can use Cholesky to sample
                # A is positive definite (initialized as I*lam, updated with outer product)
                # Add small jitter for numerical stability
                A_reg = self.A[a] + 1e-8 * np.eye(self.d)
                
                # Sample from N(theta_hat, A^{-1})
                # We can do: theta_sample = theta_hat + A^{-1/2} z
                # Or use the fact that if L is Cholesky of A, then A^{-1} = L^{-T} L^{-1}
                # So we can solve L v = z, then theta_sample = theta_hat + L^{-T} v?
                # Actually, if A = L L^T, then A^{-1} = L^{-T} L^{-1}.
                # Let z ~ N(0, I). We want y ~ N(0, A^{-1}).
                # y = A^{-1/2} z.
                # Note that if we solve L^T w = z, then w = L^{-T} z.
                # Then y = L^{-1} w? No.
                # Let's use the standard trick:
                # Generate z ~ N(0, I).
                # Solve L v = z  => v = L^{-1} z
                # Then y = L^{-T} v = L^{-T} L^{-1} z = A^{-1} z. This is not N(0, A^{-1}).
                # Wait. If z ~ N(0, I), then A^{-1/2} z ~ N(0, A^{-1}).
                # How to compute A^{-1/2} z efficiently?
                # Let A = L L^T. Then A^{-1/2} = L^{-T} (L L^T)^{-1/2} L^T? No.
                # Standard method:
                # 1. Cholesky decomposition A = L L^T.
                # 2. Generate z ~ N(0, I).
                # 3. Solve L^T w = z. (w = L^{-T} z)
                # 4. Solve L v = w. (v = L^{-1} w = L^{-1} L^{-T} z = A^{-1} z)
                # This gives a vector with covariance A^{-1} A^{-1}? No.
                # Let's check: Cov(v) = E[vv^T] = A^{-1} E[zz^T] A^{-T} = A^{-1} I A^{-1} = A^{-2}.
                # That's wrong. We want Cov = A^{-1}.
                
                # Correct method for sampling from N(mu, Sigma) where Sigma = A^{-1}:
                # We want y such that Cov(y) = A^{-1}.
                # Let A = L L^T. Then A^{-1} = L^{-T} L^{-1}.
                # We can write A^{-1} = (L^{-T}) (L^{-1}) = (L^{-1})^T (L^{-1}).
                # So if we let M = L^{-1}, then Sigma = M^T M.
                # If z ~ N(0, I), then y = M^T z has Cov(y) = M^T M = A^{-1}.
                # So:
                # 1. Cholesky A = L L^T.
                # 2. Solve L^T u = z. (u = L^{-T} z)
                # 3. Solve L v = u. (v = L^{-1} u = L^{-1} L^{-T} z)
                # Wait, M = L^{-1}. We want y = M^T z = L^{-T} z.
                # So we just need to solve L^T y = z.
                # Let's verify: y = L^{-T} z. Cov(y) = L^{-T} E[zz^T] L^{-1} = L^{-T} L^{-1} = A^{-1}.
                # Yes! So we just need to solve L^T y = z.
                
                L = np.linalg.cholesky(A_reg)
                z = self.rng.normal(0, 1, self.d)
                # Solve L^T y = z
                y = np.linalg.solve(L.T, z)
                
                theta_sample = theta_hat + y
                
                mean = theta_sample @ context
                scores.append(mean)
                
            except np.linalg.LinAlgError:
                # Fallback to mean if Cholesky fails
                try:
                    theta_hat = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta_hat @ context)
                except:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        
        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.array(context, dtype=float)
        
        # Adapt forgetting factor
        self._adapt_forgetting(reward)
        
        # Update with forgetting factor
        # A_new = beta * A_old + x x^T
        # b_new = beta * b_old + r x
        self.A[arm] = self.beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.beta * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
