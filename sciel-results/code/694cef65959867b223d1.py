import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    T = np.ones((n, m, p))
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 140
    
    def compute_error(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.max(np.abs(result - T))
    
    def verify_half_integer(U, V, W, rank):
        U2 = np.round(U * 2)
        V2 = np.round(V * 2)
        W2 = np.round(W * 2)
        
        if not np.allclose(U2, U * 2, atol=1e-6):
            return False
        if not np.allclose(V2, V * 2, atol=1e-6):
            return False
        if not np.allclose(W2, W * 2, atol=1e-6):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(int(U2[i * m + j, r]), 2) * \
                                   Fraction(int(V2[j * p + k, r]), 2) * \
                                   Fraction(int(W2[k * n + i, r]), 2)
                    if total != 1:
                        return False
        return True
    
    for target_rank in [14, 15, 16, 17]:
        if time.time() - start_time > deadline:
            break
        
        found = False
        max_restarts = 100 if target_rank <= 15 else 50
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            
            rng = np.random.RandomState(restart * 7919 + target_rank * 104729 + 1)
            
            # Diverse initialization
            strategy = restart % 4
            if strategy == 0:
                scale = 0.2 + 0.1 * (restart % 5)
                U = rng.randn(n * m, target_rank) * scale
                V = rng.randn(m * p, target_rank) * scale
                W = rng.randn(p * n, target_rank) * scale
            elif strategy == 1:
                U = rng.randint(0, 2, (n * m, target_rank)).astype(float) * 0.5
                V = rng.randint(0, 2, (m * p, target_rank)).astype(float) * 0.5
                W = rng.randint(0, 2, (p * n, target_rank)).astype(float) * 0.5
            elif strategy == 2:
                U = rng.choice([0, 0.5, 1, 1.5, 2], (n * m, target_rank))
                V = rng.choice([0, 0.5, 1, 1.5, 2], (m * p, target_rank))
                W = rng.choice([0, 0.5, 1, 1.5, 2], (p * n, target_rank))
            else:
                # Sparse structured
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    idx_u = rng.randint(0, n * m)
                    idx_v = rng.randint(0, m * p)
                    idx_w = rng.randint(0, p * n)
                    U[idx_u, r] = rng.choice([0.5, 1, 1.5])
                    V[idx_v, r] = rng.choice([0.5, 1, 1.5])
                    W[idx_w, r] = rng.choice([0.5, 1, 1.5])
            
            prev_error = np.inf
            for iter_ in range(300):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((target_rank, p))
                        for k in range(p):
                            A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U_new[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                U = U_new
                
                # Update V
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((target_rank, n))
                        for i in range(n):
                            A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V_new[j * p + k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                V = V_new
                
                # Update W
                W_new = W.copy()
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((target_rank, m))
                        for j in range(m):
                            A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W_new[k * n + i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                W = W_new
                
                error = compute_error(U, V, W, target_rank)
                
                if error < 1e-8:
                    break
                if error > prev_error * 1.05 and iter_ > 30:
                    break
                prev_error = error
            
            if prev_error < 1e-6:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify_half_integer(U_rounded, V_rounded, W_rounded, target_rank):
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    if best_rank == R_school:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
