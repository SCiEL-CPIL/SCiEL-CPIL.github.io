import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 330s limit, use 300s for safety
        start_time = time.time()
        deadline = start_time + 300.0

        D = self.dim
        # Standard CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(D))
        # Initial sigma
        sigma = 1.0
        # Initial mean
        x = np.zeros(D)
        # Covariance matrix
        C = np.eye(D)
        # Inverse covariance matrix (for efficient computation)
        C_inv = np.eye(D)
        # Eigendecomposition of C
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Step size control parameters
        mu = lam // 2  # Number of selected points
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (D + mu_eff + 5)
        c_c = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        c_1 = 2 / ((D + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2)**2 + mu_eff))
        
        # Memory for path updates
        p_sigma = np.zeros(D)
        p_c = np.zeros(D)
        # Initial condition number
        chi_N = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D**2))
        
        # Best solution found so far
        best_x = x
        best_f = func(best_x)
        calls_used = 1
        
        # Main loop
        while calls_used < self.budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate offspring
            ys = np.zeros((lam, D))
            fs = np.zeros(lam)
            
            # Sample from multivariate normal
            # We use the eigendecomposition for efficiency and stability
            # z ~ N(0, I)
            zs = np.random.normal(0, 1, size=(lam, D))
            # y = x + sigma * (eigvecs @ (eigvals**0.5 * zs.T)).T
            # This is equivalent to x + sigma * C^{1/2} z
            # Note: eigvecs are columns of eigenvectors
            scaled_zs = zs * np.sqrt(eigvals)
            # Transform to physical space
            # zs is (lam, D), we want to multiply by C^{1/2}
            # C^{1/2} = eigvecs @ diag(sqrt(eigvals))
            # So y_i = x + sigma * (eigvecs @ (sqrt(eigvals) * z_i.T)).T
            # Vectorized: Y = X + sigma * (Z * sqrt(eigvals)) @ eigvecs.T
            # Wait, z is row vector. 
            # Let's stick to the standard update:
            # y = x + sigma * sum_j z_j * eigvecs[:, j] * sqrt(eigvals[j])
            
            # Efficiently:
            # ys = x + sigma * (zs * np.sqrt(eigvals)) @ eigvecs.T
            ys = x + sigma * (zs * np.sqrt(eigvals)) @ eigvecs.T
            
            # Evaluate
            for i in range(lam):
                if calls_used >= self.budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                # Clip to box [-5, 5]
                y_i = np.clip(ys[i], -5, 5)
                f_i = func(y_i)
                fs[i] = f_i
                calls_used += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = y_i
            
            if calls_used >= self.budget or time.time() > deadline - 5.0:
                break

            # Sort by fitness
            order = np.argsort(fs)
            fs = fs[order]
            ys = ys[order]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update path p_sigma
            # C^{1/2} is eigvecs @ diag(sqrt(eigvals))
            # We need C^{-1/2} (y - x)
            # C^{-1/2} = diag(1/sqrt(eigvals)) @ eigvecs.T
            # Let's compute the difference in the eigen-space
            # d = y - x
            # d_eig = eigvecs.T @ d
            # d_norm = d_eig / sqrt(eigvals)
            
            # For efficiency, we can update p_sigma in the current coordinate system
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} (y_w - x)
            
            # Calculate C^{-1/2} (y_w - x)
            # y_w is the weighted average of the best mu solutions
            # Actually, the standard formula uses the mean of the selected points.
            # Let's compute the vector v = x_new - x
            # v_eig = eigvecs.T @ v
            # v_norm = v_eig / sqrt(eigvals)
            
            v = x_new - x
            v_eig = eigvecs.T @ v
            v_norm = v_eig / np.sqrt(eigvals)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_norm
            
            # Update path p_c
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * C^{-1/2} (y_w - x)
            # Wait, the standard update for p_c is:
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * C^{-1/2} (y_w - x)
            # But often it's defined with respect to the mean shift.
            # Let's use the standard CMA-ES update rules.
            
            # Actually, a common implementation updates p_c as:
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (C^{-1/2} * (x_new - x))
            # This is the same term as v_norm.
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * v_norm
            
            # Update covariance matrix
            # C = (1 + c_1 + c_mu - c_1 * D) C + c_1 * (p_c p_c^T - C) + c_mu * sum w_i (y_i - x)(y_i - x)^T
            # Note: The term (1 + c_1 + c_mu - c_1 * D) ensures trace preservation approximately.
            
            # Compute the outer products
            # We need to transform the differences back to the eigen-space or update in physical space?
            # Updating in physical space is easier for code clarity, though eigen-space is faster.
            # Given D=5, physical space is fine.
            
            # Term 1: p_c p_c^T
            # We have p_c in "normalized" space? No, p_c is usually maintained in the coordinate system of C.
            # Let's stick to the standard algorithm which maintains p_sigma and p_c in the "C^{-1/2}" transformed space?
            # No, standard CMA-ES maintains p_sigma and p_c in the original space but scaled?
            # Let's look at a reliable reference implementation structure.
            
            # Reference: Hansen, "The CMA Evolution Strategy: A Comparing Review"
            # p_sigma is updated with C^{-1/2} (y_w - x)
            # p_c is updated with C^{-1/2} (y_w - x)
            # The covariance update is:
            # C = (1 + c_1 (2 - h_sigma) + c_mu - c_1 D) C + c_1 (h_sigma p_sigma p_sigma^T - C) + c_mu sum w_i (y_i - x)(y_i - x)^T
            # Wait, p_sigma and p_c are vectors in the D-dimensional space.
            # If we maintain them in the "normalized" space (i.e., C^{-1/2} v), then we need to transform them back to physical space for the outer product in the C update?
            # Actually, it's easier to maintain p_sigma and p_c in the physical space?
            # No, the standard update for C uses the outer product of the difference vector.
            
            # Let's use a simpler, robust update that works well for small D:
            # 1. Compute the weighted sum of outer products of (y_i - x) for the best mu points.
            # 2. Update C.
            
            # Let's re-evaluate the p_sigma and p_c updates.
            # In many implementations, p_sigma and p_c are stored in the same space as x.
            # However, the term C^{-1/2} (y_w - x) requires the inverse square root.
            
            # Let's assume p_sigma and p_c are in the physical space.
            # Then we need to compute C^{-1/2} (x_new - x).
            # We have v_norm = C^{-1/2} (x_new - x).
            # So the update is:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(...) * v_norm
            # This implies p_sigma is in the "normalized" space?
            # If p_sigma is in normalized space, then p_sigma p_sigma^T is in normalized space.
            # But C is in physical space.
            
            # Correct approach:
            # Maintain p_sigma and p_c in the physical space?
            # No, the standard CMA-ES algorithm maintains the paths in the space where the step is taken.
            # Actually, let's look at the `cma` package source or a standard tutorial.
            # 
            # Standard Update:
            # 1. Compute y_w = sum w_i y_i
            # 2. p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) C^{-1/2} (y_w - x)
            # 3. h_sigma = ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2 iter}) / chi_N
            # 4. p_c = (1 - c_c) p_c + h_sigma sqrt(c_c (2 - c_c) mu_eff) C^{-1/2} (y_w - x)
            # 5. C = (1 + c_1 (2 - h_sigma) - c_1 D + c_mu) C + c_1 (h_sigma p_sigma p_sigma^T - C) + c_mu sum w_i (y_i - x)(y_i - x)^T
            #
            # Notice that p_sigma and p_c are in the "C^{-1/2}" transformed space?
            # No, the term C^{-1/2} (y_w - x) is a vector. If we add it to p_sigma, p_sigma must be in the same space.
            # However, the term p_sigma p_sigma^T appears in the update of C.
            # If p_sigma is in the "normalized" space, then p_sigma p_sigma^T is in the "normalized" space.
            # But C is in physical space.
            # This suggests that p_sigma and p_c are actually maintained in the physical space, but the update term is transformed?
            # No, the term is C^{-1/2} (y_w - x). Let's call this vector u.
            # If p_sigma is in physical space, adding u (which is in "normalized" space?) doesn't make sense dimensionally unless we transform u back.
            #
            # Let's clarify:
            # Let z = C^{-1/2} (y_w - x). This z is a vector in R^D.
            # The update is p_sigma = (1 - c_sigma) p_sigma + alpha * z.
            # This implies p_sigma is in the same space as z.
            # What space is z in?
            # If C = I, z = y_w - x.
            # If C is not I, z is the "whitened" difference.
            # The outer product p_sigma p_sigma^T is then in the "whitened" space.
            # But the update for C is:
            # C_new = ... + c_1 (h_sigma p_sigma p_sigma^T - C) ...
            # This implies that p_sigma p_sigma^T must be in the same space as C.
            # This is a contradiction unless p_sigma is in the physical space.
            #
            # Let's check the literature.
            # Hansen, "The CMA Evolution Strategy: A Comparing Review" (2016)
            # Algorithm 1:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) C^{-1/2} (y_w - x)
            # ...
            # C = (1 + c_1 (2 - h_sigma) - c_1 D + c_mu) C + c_1 (h_sigma p_sigma p_sigma^T - C) + ...
            #
            # It seems p_sigma is indeed in the "C^{-1/2}" transformed space?
            # But then p_sigma p_sigma^T is in that space.
            # How can we add it to C (physical space)?
            # Ah, the term is actually C^{1/2} p_sigma p_sigma^T C^{1/2}?
            # No, the formula is usually written as:
            # C = ... + c_1 (h_sigma C^{1/2} p_sigma p_sigma^T C^{1/2} - C) ...
            #
            # To avoid confusion and potential bugs in the transformation logic, and given D=5 is very small,
            # I will use a simpler, more direct implementation that updates C directly using the outer products of the differences in physical space,
            # and uses a simpler step size control.
            #
            # Actually, a very robust and simple alternative for small D is to use a **Separable CMA-ES** or even just a well-tuned **(1+lambda)-ES with adaptive step size and rotation**.
            #
            # However, let's try to implement the standard CMA-ES correctly.
            # The key is that p_sigma and p_c are vectors in R^D.
            # The update for C involves the outer product of these vectors.
            # If we maintain p_sigma in the "physical" space, the update term should be the outer product of the physical difference.
            # But the standard algorithm uses the whitened difference to control the step size and path memory.
            #
            # Let's use a library-like implementation found in common Python snippets for CMA-ES.
            #
            # Re-implementing with careful attention to the space of the vectors:
            # We will maintain p_sigma and p_c in the "eigen" space of the current C?
            # No, that changes every iteration.
            #
            # Let's stick to the physical space for p_sigma and p_c, but compute the update terms using the eigendecomposition.
            #
            # Update:
            # 1. Compute v = x_new - x
            # 2. Compute v_eig = eigvecs.T @ v
            # 3. Compute v_norm = v_eig / sqrt(eigvals)
            # 4. p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_norm
            #    (Here p_sigma is in the "normalized" space)
            # 5. Compute norm_p_sigma = np.linalg.norm(p_sigma)
            # 6. h_sigma = norm_p_sigma / (chi_N * np.sqrt(1 - (1 - c_sigma)**(2 * iter)))
            #    (We need to track iter or use an approximation. For small budgets, we can approximate the denominator as 1 if c_sigma is small, or track it.)
            #    Let's track the cumulative factor.
            # 7. p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * v_norm
            #    (Here p_c is in the "normalized" space)
            # 8. Update C:
            #    We need to form the outer product of p_sigma and p_c in the physical space.
            #    p_sigma_phys = eigvecs @ p_sigma
            #    p_c_phys = eigvecs @ p_c
            #    
            #    C = (1 + c_1 * (2 - h_sigma) - c_1 * D + c_mu) * C 
            #        + c_1 * (h_sigma * np.outer(p_sigma_phys, p_sigma_phys) - C)
            #        + c_mu * sum(weights[i] * np.outer(ys[i] - x, ys[i] - x) for i in range(mu))
            #
            # 9. Update eigendecomposition of C.
            #    C = eigvecs @ diag(eigvals) @ eigvecs.T
            #    We can update eigvals and eigvecs using the eigenvalues of the update matrix?
            #    Or just recompute eigendecomposition. D=5, so np.linalg.eigh is fast.
            
            # Let's implement this version.
            
            # Compute h_sigma
            # We need the history factor.
            # hist_sigma = 1 - (1 - c_sigma)**(2 * iter)
            # Since we don't track iter explicitly in a simple loop, we can approximate.
            # For small iter, (1 - c_sigma)**(2 * iter) is close to 1 - 2 * c_sigma * iter.
            # Let's just use the norm directly and assume h_sigma is 1 if the norm is reasonable, or use a simple heuristic.
            # Actually, for a robust simple implementation, we can skip the h_sigma heuristic and just use the norm.
            # Or, we can maintain a counter.
            
            iter_count = 0
            # Initialize history factors
            hist_sigma = 1.0
            hist_c = 1.0
            
            # We are inside the loop now, so we need to do the updates.
            # Let's move the update logic here.
            
            # 1. Compute v and v_norm
            v = x_new - x
            v_eig = eigvecs.T @ v
            v_norm = v_eig / np.sqrt(eigvals)
            
            # 2. Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_norm
            
            # 3. Compute h_sigma
            # hist_sigma is updated each step: hist_sigma = 1 - (1 - c_sigma) * hist_sigma
            hist_sigma = 1 - (1 - c_sigma) * hist_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = norm_p_sigma / (chi_N * np.sqrt(hist_sigma))
            
            # 4. Update p_c
            # hist_c is updated: hist_c = 1 - (1 - c_c) * hist_c
            hist_c = 1 - (1 - c_c) * hist_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_norm
            
            # 5. Update C
            # Transform p_sigma and p_c back to physical space
            p_sigma_phys = eigvecs @ p_sigma
            p_c_phys = eigvecs @ p_c
            
            # Compute the sum of outer products for the best mu points
            sum_outer = np.zeros((D, D))
            for i in range(mu):
                diff = ys[i] - x
                sum_outer += weights[i] * np.outer(diff, diff)
            
            # Update C
            # Note: The term (1 + c_1 (2 - h_sigma) - c_1 D + c_mu) is the decay factor
            decay = 1 + c_1 * (2 - h_sigma) - c_1 * D + c_mu
            C = decay * C + c_1 * (h_sigma * np.outer(p_sigma_phys, p_sigma_phys) - C) + c_mu * sum_outer
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # 6. Update eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure eigvals are positive
            eigvals = np.maximum(eigvals, 1e-12)
            
            # 7. Update x
            x = x_new
            
            # 8. Adapt sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_N - 1))
            # d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (D + 1) - 1) + c_sigma
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (D + 1) - 1) + c_sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
            
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-8, 1e8)

        return best_x, best_f
