import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if indices match, else 0
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    rank_school = n * m * p
    
    # Fallback: Schoolbook
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Heuristic Search: ALS + Rounding
    # Try ranks from 41 down to 35
    # 41 is the theoretical bound from 23+18
    for target_rank in range(41, 34, -1):
        found = False
        # Try multiple random initializations
        for _ in range(100):
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS Optimization
            # We minimize || T - sum(U_r V_r W_r) ||^2
            # Fix V, W -> Solve for U: U = (sum_r T_{:, :, r} W_r^T) (sum_r V_r V_r^T)^{-1}?
            # Actually, for each r, U_r is a vector.
            # The term is outer(V_r, W_r) contracted with T?
            # T = sum_r U_r \otimes V_r \otimes W_r
            # Flatten T to matrix T_{alpha, (beta, gamma)}
            # T_{alpha, beta gamma} = sum_r U_{alpha r} (V_{beta r} W_{gamma r})
            # Let M_r = V_r \otimes W_r (vector of length m*p * p*n)
            # Then T_{alpha, :} = sum_r U_{alpha r} M_r^T
            # So [T]_{alpha, beta gamma} = U_{alpha r} M_{r, beta gamma}
            # This is a linear system for U if V, W are fixed.
            
            # Reshape T to (n*m, m*p*p*n)
            T_flat = T.reshape(n*m, m*p * p*n)
            
            # Precompute M matrix: (m*p*p*n, rank)
            # M[:, r] = V[:, r] \otimes W[:, r]
            # V is (m*p, rank), W is (p*n, rank)
            # We need to compute Kronecker product for each column?
            # Or use einsum.
            # M_{beta gamma, r} = V_{beta r} W_{gamma r}
            
            for it in range(20):
                # Solve for U
                # M (size (m*p*p*n, rank))
                M = np.einsum('br,cr->bc r', V, W) # This is wrong shape.
                # V is (m*p, rank), W is (p*n, rank)
                # We want M_{beta, gamma, r} = V_{beta, r} * W_{gamma, r}
                # Reshape to (m*p * p*n, rank)
                M = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(-1, target_rank)
                
                # T_flat is (n*m, m*p*p*n)
                # We want U (n*m, rank) such that T_flat approx U @ M.T
                # T_flat = U @ M.T  => U = T_flat @ M @ (M.T @ M)^-1
                # Use least squares
                try:
                    U, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                    U = U.T
                except:
                    pass

                # Solve for V
                # T_{alpha, beta, gamma} = sum_r U_{alpha r} V_{beta r} W_{gamma r}
                # Fix U, W.
                # For each beta, T_{alpha, beta, gamma} = sum_r V_{beta r} (U_{alpha r} W_{gamma r})
                # Let N_{alpha, gamma, r} = U_{alpha r} W_{gamma r}
                # T_{alpha, beta, gamma} = sum_r V_{beta r} N_{alpha, gamma, r}
                # Reshape T to (m*p, n*m*p*n)
                T_flat_V = T.reshape(m*p, n*m * p*n)
                N = (U[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(-1, target_rank)
                # N is (n*m*p*n, rank)
                # T_flat_V (m*p, n*m*p*n) = V (m*p, rank) @ N.T (rank, n*m*p*n)
                try:
                    V, _, _, _ = np.linalg.lstsq(N.T, T_flat_V.T, rcond=None)
                    V = V.T
                except:
                    pass

                # Solve for W
                # T_{alpha, beta, gamma} = sum_r W_{gamma r} (U_{alpha r} V_{beta r})
                # Fix U, V.
                # Let P_{alpha, beta, r} = U_{alpha r} V_{beta r}
                # T_{alpha, beta, gamma} = sum_r W_{gamma r} P_{alpha, beta, r}
                # Reshape T to (p*n, n*m*m*p)
                T_flat_W = T.reshape(p*n, n*m * m*p)
                P = (U[:, :, np.newaxis] * V[np.newaxis, :, :]).reshape(-1, target_rank)
                # P is (n*m*m*p, rank)
                # T_flat_W (p*n, n*m*m*p) = W (p*n, rank) @ P.T
                try:
                    W, _, _, _ = np.linalg.lstsq(P.T, T_flat_W.T, rcond=None)
                    W = W.T
                except:
                    pass

            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            # Reconstruct T_check
            T_check = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_check += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
            
            if np.allclose(T_check, T, atol=1e-9):
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = target_rank
                found = True
                break
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
