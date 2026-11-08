import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build target tensor for validation
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1
    
    def validate(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if decomposition exactly matches target tensor"""
        if U_mat.shape[1] != V_mat.shape[1] or U_mat.shape[1] != W_mat.shape[1]:
            return False
        reconstructed = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(reconstructed, target, atol=tol)
    
    # Start with schoolbook rank 27
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank by finding linear dependencies among terms
    # Strategy: for each pair of terms, check if they can be merged
    # A better strategy: use local search to zero out entries
    
    # Try rank 25: start from schoolbook and try to eliminate 2 terms
    # by finding linear combinations
    
    # Approach: use the fact that we can sometimes express products using
    # fewer multiplications by exploiting algebraic identities
    
    # For 3x3, one known approach is to use a block structure
    # Let's try a structured search for rank 25
    
    # Try all pairs of terms to see if one can be expressed as a combination
    # of others (this is expensive, so we limit it)
    
    for target_rank in [25, 24, 23]:
        if time.time() > deadline - 15:
            break
        
        # Try to find a rank-target_rank decomposition
        # Strategy: random initialization + local refinement
        max_attempts = 2000 if target_rank >= 24 else 5000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U_try = np.random.randn(n * m, target_rank) * 0.5
            V_try = np.random.randn(m * p, target_rank) * 0.5
            W_try = np.random.randn(p * n, target_rank) * 0.5
            
            # Try a few local optimization steps
            # Use gradient-free approach: perturb and check
            best_err = np.inf
            for step in range(50):
                if time.time() > deadline - 8:
                    break
                
                # Compute current error
                recon = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
                err = np.linalg.norm(recon - target)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        found = True
                        break
                
                # Random perturbation
                noise_scale = 0.1 if step < 20 else 0.01
                U_try += np.random.randn(n * m, target_rank) * noise_scale
                V_try += np.random.randn(m * p, target_rank) * noise_scale
                W_try += np.random.randn(p * n, target_rank) * noise_scale
            
            if found:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                break
        
        if found:
            break
    
    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
