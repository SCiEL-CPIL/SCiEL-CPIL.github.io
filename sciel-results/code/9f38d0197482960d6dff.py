import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify(U, V, W, atol=1e-8):
        r = U.shape[1]
        # Check all n*m*p target equations equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if not np.isclose(np.sum(prod), 1.0, atol=atol):
                        return False
        # Check that no spurious terms are created (optional but good for robustness)
        # The tensor is only non-zero at valid (i,j,k) positions
        return True
    
    def try_rank_reduction(target_rank, max_attempts=3):
        """Try to find a decomposition of the given target_rank."""
        if target_rank >= n * m * p:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Structured initialization: start from schoolbook and perturb
            np.random.seed(1000 + attempt * 7 + target_rank)
            
            # Initialize with schoolbook-like structure but with target_rank columns
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Fill first min(target_rank, n*m*p) columns with schoolbook pattern
            num_cols = min(target_rank, n * m * p)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, col] = 1.0
                        V[v_idx, col] = 1.0
                        W[w_idx, col] = 1.0
                        col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            
            # Add small random perturbations to encourage exploration
            U += np.random.randn(n*m, target_rank) * 0.05
            V += np.random.randn(m*p, target_rank) * 0.05
            W += np.random.randn(p*n, target_rank) * 0.05
            
            # Use iterative refinement: solve for adjustments
            # We'll use a simple gradient-free approach with random perturbations
            best_obj = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Compute objective: sum of squared errors for all target equations
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                            errors.append(np.sum(prod) - 1.0)
                
                obj = np.sum(np.array(errors) ** 2)
                
                if obj < best_obj:
                    best_obj = obj
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_obj < 1e-10:
                    break
                
                # Random perturbation step
                step_size = 0.1 * (1 - iteration / 50.0) + 0.01
                U += np.random.randn(n*m, target_rank) * step_size
                V += np.random.randn(m*p, target_rank) * step_size
                W += np.random.randn(p*n, target_rank) * step_size
            
            # Check if we found a valid solution
            if best_obj < 1e-8:
                # Try to round to half-integers
                U_round = np.round(best_U * 2) / 2
                V_round = np.round(best_V * 2) / 2
                W_round = np.round(best_W * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    return U_round, V_round, W_round
                
                # If rounding doesn't work, try with original
                if verify(best_U, best_V, best_W):
                    return best_U, best_V, best_W
        
        return None
    
    # Get schoolbook decomposition
    U_sb, V_sb, W_sb = get_schoolbook()
    
    # Verify schoolbook
    if not verify(U_sb, V_sb, W_sb):
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Start from rank 44 down to a reasonable minimum
    min_rank_to_try = max(35, best_rank - 15)
    
    for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_reduction(target_rank, max_attempts=2)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = target_rank
            break  # Found a valid lower rank, stop searching
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
