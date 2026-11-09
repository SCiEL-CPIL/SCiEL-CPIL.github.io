import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 5
    nm = n * m   # 8
    mp = m * p   # 20
    pn = p * n   # 10
    
    start_time = time.time()
    deadline = start_time + 135  # safety margin
    
    # Build the target tensor T[a,b,c]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 40)
    U_sb = np.zeros((nm, nm * mp * pn))
    V_sb = np.zeros((mp, nm * mp * pn))
    W_sb = np.zeros((pn, nm * mp * pn))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-8):
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero entries (sample if too many)
        total_zeros = nm * mp * pn - n * m * p
        if total_zeros <= 2000:
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        i_a = a // m
                        j_a = a % m
                        j_b = b // p
                        k_b = b % p
                        k_c = c // n
                        i_c = c % n
                        if not (i_a == i_c and j_a == j_b and k_b == k_c):
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(val) > tol:
                                return False
        else:
            # Sample zero entries
            for _ in range(500):
                a = random.randint(0, nm - 1)
                b = random.randint(0, mp - 1)
                c = random.randint(0, pn - 1)
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (i_a == i_c and j_a == j_b and k_b == k_c):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    def als_decompose(T, target_rank, init=None, max_iters=300, tol=1e-10):
        """Run ALS to find a rank-r decomposition of tensor T."""
        if init is not None:
            U, V, W = init
        else:
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Build design matrix for U updates
            # A[r_idx] = V[:,r] ⊗ W[:,r], shape (mp*pn, rank)
            A_U = np.zeros((mp * pn, target_rank))
            for r in range(target_rank):
                A_U[:, r] = (V[:, r] * W[:, r]).flatten()
            
            # Update U
            for a in range(nm):
                rhs = T[a, :, :].flatten()
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=None)
                except:
                    pass
            
            # Build design matrix for V updates
            A_V = np.zeros((nm * pn, target_rank))
            for r in range(target_rank):
                A_V[:, r] = (U[:, r] * W[:, r]).flatten()
            
            # Update V
            for b in range(mp):
                rhs = T[:, b, :].flatten()
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=None)
                except:
                    pass
            
            # Build design matrix for W updates
            A_W = np.zeros((nm * mp, target_rank))
            for r in range(target_rank):
                A_W[:, r] = (U[:, r] * V[:, r]).flatten()
            
            # Update W
            for c in range(pn):
                rhs = T[:, :, c].flatten()
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=None)
                except:
                    pass
        
        return U, V, W
    
    def polish(U, V, W, target_rank, max_iters=100, tol=1e-12):
        """High-precision polish pass with tighter tolerance."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute residual
            A_U = np.zeros((mp * pn, target_rank))
            for r in range(target_rank):
                A_U[:, r] = (V[:, r] * W[:, r]).flatten()
            
            for a in range(nm):
                rhs = T[a, :, :].flatten()
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=1e-14)
                except:
                    pass
            
            A_V = np.zeros((nm * pn, target_rank))
            for r in range(target_rank):
                A_V[:, r] = (U[:, r] * W[:, r]).flatten()
            
            for b in range(mp):
                rhs = T[:, b, :].flatten()
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=1e-14)
                except:
                    pass
            
            A_W = np.zeros((nm * mp, target_rank))
            for r in range(target_rank):
                A_W[:, r] = (U[:, r] * V[:, r]).flatten()
            
            for c in range(pn):
                rhs = T[:, :, c].flatten()
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=1e-14)
                except:
                    pass
        
        return U, V, W
    
    def round_to_halfint(mat, tol=1e-6):
        """Try to round to nearest half-integer."""
        mat_r = mat.copy()
        for i in range(mat_r.shape[0]):
            for j in range(mat_r.shape[1]):
                val = mat_r[i, j]
                # Try half-integers
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < tol:
                    mat_r[i, j] = rounded
                else:
                    rounded_int = round(val)
                    if abs(val - rounded_int) < tol:
                        mat_r[i, j] = float(rounded_int)
        return mat_r
    
    # Try ranks from just below schoolbook downward
    best_rank = nm * mp * pn
    best_result = None
    
    # Use schoolbook as initialization for truncated ranks
    for target_rank in range(39, 14, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Multiple random restarts for lower ranks
        n_restarts = 3 if target_rank >= 30 else 2 if target_rank >= 20 else 1
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            if restart == 0 and target_rank <= nm * mp * pn:
                # Initialize from schoolbook (truncated)
                U_init = U_sb[:, :target_rank].copy()
                V_init = V_sb[:, :target_rank].copy()
                W_init = W_sb[:, :target_rank].copy()
            else:
                # Random initialization
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
            
            U_curr, V_curr, W_curr = als_decompose(T, target_rank, init=(U_init, V_init, W_init), max_iters=200)
            
            # Polish
            U_curr, V_curr, W_curr = polish(U_curr, V_curr, W_curr, target_rank, max_iters=50)
            
            if verify(U_curr, V_curr, W_curr):
                # Try to round
                U_r = round_to_halfint(U_curr)
                V_r = round_to_halfint(V_curr)
                W_r = round_to_halfint(W_curr)
                
                if verify(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_curr.tolist()}
        
        if found:
            break
    
    # Fallback: return schoolbook
    U_out = U_sb[:, :nm*mp*pn]
    V_out = V_sb[:, :nm*mp*pn]
    W_out = W_sb[:, :nm*mp*pn]
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
