"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    deadline = time.time() + 135
    
    # Try to find lower rank via structured search
    # Use the fact that for fixed (V,W), U is determined by linear system
    # and vice versa
    
    rng = np.random.RandomState(123)
    
    # Try ranks from 53 down
    for target_rank in range(rank_sb - 1, 25, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # Alternating minimization
            for it in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U given V, W
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                M_flat = np.einsum('br,cr->rbc', V, W).reshape(target_rank, m * p * p * n)
                T_a = T.reshape(n * m, m * p * p * n)
                try:
                    A = M_flat @ M_flat.T + 1e-6 * np.eye(target_rank)
                    U = T_a @ M_flat.T @ np.linalg.inv(A)
                except:
                    break
                
                # Solve for V given U, W
                N_flat = np.einsum('ar,cr->rac', U, W).reshape(target_rank, n * m * p * n)
                T_b = T.reshape(m * p, n * m * p * n)
                try:
                    B = N_flat @ N_flat.T + 1e-6 * np.eye(target_rank)
                    V = T_b @ N_flat.T @ np.linalg.inv(B)
                except:
                    break
                
                # Solve for W given U, V
                P_flat = np.einsum('ar,br->rab', U, V).reshape(target_rank, n * m * m * p)
                T_c = T.reshape(p * n, n * m * m * p)
                try:
                    C = P_flat @ P_flat.T + 1e-6 * np.eye(target_rank)
                    W = T_c @ P_flat.T @ np.linalg.inv(C)
                except:
                    break
                
                if it % 50 == 0:
                    # Quick error check
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a, b, c = i*m+j, j*p+k, k*n+i
                                val = np.dot(U[a], V[b] * W[c])
                                err += (val - 1.0)**2
                    if err < 1e-12:
                        found = True
                        break
            
            if found:
                # Full verification
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a, b, c = i*m+j, j*p+k, k*n+i
                            val = np.dot(U[a], V[b] * W[c])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
        
        if best_rank < rank_sb:
            continue
        else:
            break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
