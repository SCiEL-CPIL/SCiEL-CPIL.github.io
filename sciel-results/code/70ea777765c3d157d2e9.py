import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build the rank-36 Strassen-based decomposition
    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def u_idx(i, j):
            return i * m + j
        
        def v_idx(j, k):
            return j * p + k
        
        def w_idx(k, i):
            return k * n + i
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        U[u_idx(0,0)][col] += 1
        V[v_idx(0,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        
        U[u_idx(0,1)][col] += 1
        V[v_idx(1,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        
        U[u_idx(1,0)][col] += 1
        V[v_idx(0,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        U[u_idx(1,1)][col] += 1
        V[v_idx(1,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        U[u_idx(0,2)][col] += 1
        V[v_idx(2,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        
        U[u_idx(0,3)][col] += 1
        V[v_idx(3,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        
        U[u_idx(1,2)][col] += 1
        V[v_idx(2,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        U[u_idx(1,3)][col] += 1
        V[v_idx(3,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return {"U": U, "V": V, "W": W}
    
    best_result = build_strassen_36()
    best_rank = 36
    
    U0 = np.array(best_result['U'], dtype=float)
    V0 = np.array(best_result['V'], dtype=float)
    W0 = np.array(best_result['W'], dtype=float)
    
    # Pre-compute target and non-target indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    all_indices = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    non_target_indices = [idx for idx in all_indices if idx not in set(target_indices)]
    
    def residuals(params, rank):
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u + num_v]
        W_flat = params[num_u + num_v:]
        
        U = U_flat.reshape((n * m, rank))
        V = V_flat.reshape((m * p, rank))
        W = W_flat.reshape((p * n, rank))
        
        res = []
        for a, b, c in target_indices:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            res.append(val - 1.0)
        for a, b, c in non_target_indices:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            res.append(val)
            
        return np.array(res)
    
    # Try to find rank 35 solution
    for rank in [35, 34, 33]:
        if time.time() > deadline:
            break
            
        # Try dropping columns from the 36-rank solution
        for drop_idx in range(36):
            if time.time() > deadline:
                break
                
            U_init = np.delete(U0, drop_idx, axis=1).copy()
            V_init = np.delete(V0, drop_idx, axis=1).copy()
            W_init = np.delete(W0, drop_idx, axis=1).copy()
            
            # Pad to rank if needed (if rank < 35)
            if U_init.shape[1] < rank:
                pad_size = rank - U_init.shape[1]
                U_init = np.hstack([U_init, np.zeros((U_init.shape[0], pad_size))])
                V_init = np.hstack([V_init, np.zeros((V_init.shape[0], pad_size))])
                W_init = np.hstack([W_init, np.zeros((W_init.shape[0], pad_size))])
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(residuals, params0, args=(rank,), method='trf', max_nfev=1000)
                if res.cost < 1e-8:
                    num_u = n * m * rank
                    num_v = m * p * rank
                    num_w = p * n * rank
                    
                    U_sol = res.x[:num_u].reshape((n * m, rank))
                    V_sol = res.x[num_u:num_u + num_v].reshape((m * p, rank))
                    W_sol = res.x[num_u + num_v:].reshape((p * n, rank))
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify
                    if np.max(np.abs(residuals(np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()]), rank))) < 1e-8:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        
            except Exception:
                pass
        
        # Try random initializations
        for _ in range(10):
            if time.time() > deadline:
                break
                
            U_init = np.random.normal(0, 1, (n*m, rank))
            V_init = np.random.normal(0, 1, (m*p, rank))
            W_init = np.random.normal(0, 1, (p*n, rank))
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(residuals, params0, args=(rank,), method='trf', max_nfev=1000)
                if res.cost < 1e-8:
                    num_u = n * m * rank
                    num_v = m * p * rank
                    num_w = p * n * rank
                    
                    U_sol = res.x[:num_u].reshape((n * m, rank))
                    V_sol = res.x[num_u:num_u + num_v].reshape((m * p, rank))
                    W_sol = res.x[num_u + num_v:].reshape((p * n, rank))
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if np.max(np.abs(residuals(np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()]), rank))) < 1e-8:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        
            except Exception:
                pass
                
    return best_result
