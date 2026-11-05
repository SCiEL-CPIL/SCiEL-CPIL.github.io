"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Flatten T for each c
    T_flat_by_c = np.zeros((p * n, n * m * m * p))
    for c in range(p * n):
        T_flat_by_c[c] = T[:, :, c].flatten()
    
    def verify(U, V, W):
        """Verify exact decomposition with integer/half-integer entries."""
        # Snap to nearest half-integer
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        return np.allclose(T_recon, T, atol=1e-9, rtol=1e-9)
    
    def try_rank_R(R, max_attempts=50):
        nonlocal best_U, best_V, best_W, best_rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return False
            
            # Generate random U, V with small integer entries
            # Use -1, 0, 1 for simplicity
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Build the linear system for W
            # For each c, we need: sum_r W[c,r] * (U[:,r] ⊗ V[:,r])[idx] = T_flat_by_c[c][idx]
            # where idx corresponds to (a,b)
            
            # Precompute kron vectors
            kron_vecs = np.zeros((R, n * m * m * p))
            for r in range(R):
                kron_vecs[r] = np.kron(U[:, r], V[:, r])
            
            # For each c, solve: kron_vecs.T @ W_c = T_flat_by_c[c]
            W = np.zeros((p * n, R))
            valid = True
            for c in range(p * n):
                A = kron_vecs.T  # shape: (n*m*m*p, R)
                b = T_flat_by_c[c]
                try:
                    W_c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[c] = W_c
                    if len(residuals) > 0 and residuals[0] > 1e-8:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if verify(U, V, W):
                best_U = np.round(U * 2) / 2
                best_V = np.round(V * 2) / 2
                best_W = np.round(W * 2) / 2
                best_rank = R
                return True
        
        return False
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    # Try a few ranks
    for R in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        attempts = 30 if R < 40 else 15
        if try_rank_R(R, attempts):
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
