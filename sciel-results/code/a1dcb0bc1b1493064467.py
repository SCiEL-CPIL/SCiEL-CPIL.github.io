import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim = n * m * p
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, dim))
    V_sb = np.zeros((m * p, dim))
    W_sb = np.zeros((p * n, dim))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = dim
    
    start_time = time.time()
    deadline = 140.0  # 140s out of 150s
    
    # Try border rank relaxation with rational reconstruction
    # The idea: approximate the multiplication tensor with rank-R tensors
    # where R < dim, then snap to rational entries
    
    # First, try a few lower ranks using random initialization + optimization
    # We'll use a smooth approach: minimize ||T_target - sum_r u_r ⊗ v_r ⊗ w_r||^2
    # with rational snapping at the end
    
    for R in range(dim - 1, max(dim - 15, 45), -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            # Initialize with random integer vectors
            U_init = np.random.randint(-2, 3, size=(n * m, R), dtype=float)
            V_init = np.random.randint(-2, 3, size=(m * p, R), dtype=float)
            W_init = np.random.randint(-2, 3, size=(p * n, R), dtype=float)
            
            # Try to find coefficients that make this work
            # Build the tensor matrix: each rank-1 term contributes u⊗v⊗w
            # We need: sum_r coeff_r * (u_r ⊗ v_r ⊗ w_r) = T_target
            
            # Construct the linear system
            # M[:, r] = flatten(u_r ⊗ v_r ⊗ w_r)
            M = np.zeros((dim, R))
            for r in range(R):
                M[:, r] = np.einsum('i,j,k->ijk', 
                                     U_init[:, r], V_init[:, r], W_init[:, r]).flatten()
            
            t_target = np.ones((n, m, p)).flatten()
            
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(M, t_target, rcond=None)
                if residuals and residuals[0] < 1e-9:
                    # Check if we can snap coefficients to integers/halves
                    int_coeffs = np.round(coeffs * 2) / 2
                    if np.allclose(M @ int_coeffs, t_target, atol=1e-9):
                        # Found a valid solution!
                        U = U_init * int_coeffs
                        V = V_init * int_coeffs
                        W = W_init * int_coeffs
                        
                        # Snap to integers/halves
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
                        
                        # Verify
                        if _verify(U, V, W, n, m, p):
                            best_U, best_V, best_W = U, V, W
                            best_rank = R
                            # Continue searching for even lower rank
                            continue
            except:
                pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _verify(U, V, W, n, m, p):
    """Verify that the factorization is correct."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Reconstruct the tensor
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T_reconstructed += np.einsum('i,j,k->ijk', u, v, w)
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_target, atol=1e-9)
