import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 5
    
    # Target tensor: T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def validate(U, V, W):
        """Check if U, V, W exactly decompose T"""
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False
    
    def schoolbook_decomposition():
        """Return the trivial rank 45 decomposition"""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, column] = 1
                    V[j*p + k, column] = 1
                    W[k*n + i, column] = 1
                    column += 1
                    
        return U, V, W
    
    def try_alternating_minimization(rank, max_iters=500, seed=42, time_limit=20):
        """Try to find a rank-r decomposition using alternating minimization."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5
        
        b_vec = np.ones(n*m*p)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Solve for W given U, V
            # For each (i,j,k): sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
            # Unknowns: W[k*n+i, s] for k in range(p), i in range(n), s in range(rank)
            # Total unknowns: p*n*rank
            # Equations: n*m*p = 45
            
            num_unk = p * n * rank
            A_mat = np.zeros((n*m*p, num_unk))
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for s in range(rank):
                            # W[k*n+i, s] is at index (k*n+i)*rank + s
                            A_mat[eq_idx, (k*n+i)*rank + s] = U[i*m+j, s] * V[j*p+k, s]
                        eq_idx += 1
            
            W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = W_flat.reshape(p*n, rank)
            
            # Solve for U given V, W
            num_unk = n * m * rank
            A_mat = np.zeros((n*m*p, num_unk))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for s in range(rank):
                            A_mat[eq_idx, (i*m+j)*rank + s] = V[j*p+k, s] * W[k*n+i, s]
                        eq_idx += 1
            
            U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U = U_flat.reshape(n*m, rank)
            
            # Solve for V given U, W
            num_unk = m * p * rank
            A_mat = np.zeros((n*m*p, num_unk))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for s in range(rank):
                            A_mat[eq_idx, (j*p+k)*rank + s] = U[i*m+j, s] * W[k*n+i, s]
                        eq_idx += 1
            
            V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            V = V_flat.reshape(m*p, rank)
            
            # Check convergence
            if it % 50 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                if err < 1e-14:
                    break
        
        # Try rounding to integers and half-integers
        # First try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if validate(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try rounding to half-integers
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if validate(U_half, V_half, W_half):
            return U_half, V_half, W_half
        
        # Try mixed rounding: check if close to integers or half-integers
        # For each matrix, try both integer and half-integer rounding
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for U_round in [np.round(U), np.round(U * 2) / 2.0]:
            for V_round in [np.round(V), np.round(V * 2) / 2.0]:
                for W_round in [np.round(W), np.round(W * 2) / 2.0]:
                    R = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                    err = np.sum((R - T)**2)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U_round.copy(), V_round.copy(), W_round.copy()
        
        if best_err < 1e-9 and best_U is not None:
            return best_U, best_V, best_W
        
        return None
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try various ranks with multiple seeds
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But for matrix multiplication, the rank is at least n*m*p / max(n,m,p) roughly
    # Known: 3x3x3 has rank 23, 3x3x2 has rank 18 (schoolbook) or possibly lower
    
    # Try ranks from low to high
    ranks_to_try = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            result = try_alternating_minimization(rank, max_iters=300, seed=seed * 1000 + rank * 7)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W)
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, stop
            break
    
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
