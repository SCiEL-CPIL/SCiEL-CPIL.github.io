"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index triples and their positions
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_pos = np.array([i * m + j for i, j, k in indices])
    v_pos = np.array([j * p + k for i, j, k in indices])
    w_pos = np.array([k * n + i for i, j, k in indices])
    
    def verify(U, V, W, tol=1e-5):
        """Verify that the decomposition is exact using vectorized operations."""
        U_u = U[u_pos, :]
        V_v = V[v_pos, :]
        W_w = W[w_pos, :]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        max_err = np.max(np.abs(vals - 1.0))
        return max_err < tol

    def residual_func(params, rank):
        """Vectorized residual calculation."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        U_u = U_flat[u_pos, :]
        V_v = V_flat[v_pos, :]
        W_w = W_flat[w_pos, :]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        return vals - 1.0

    def get_22_term_decomp():
        """
        Construct the rank-22 block decomposition.
        14 terms from two 2x2x2 Strassen blocks.
        8 terms from two 2x1x2 Naive blocks.
        """
        U = np.zeros((nm, 22))
        V = np.zeros((mp, 22))
        W = np.zeros((pn, 22))
        
        # Helper to add Strassen 2x2x2
        def add_strassen(rank_offset, u_base, v_base, w_base):
            # m1 = (a00+a11)*(b00+b11)
            U[u_base, rank_offset+0] += 1; U[u_base+3, rank_offset+0] += 1
            V[v_base, rank_offset+0] += 1; V[v_base+3, rank_offset+0] += 1
            W[w_base, rank_offset+0] += 1; W[w_base+3, rank_offset+0] += 1
            
            # m2 = (a10+a11)*b00
            U[u_base+2, rank_offset+1] += 1; U[u_base+3, rank_offset+1] += 1
            V[v_base, rank_offset+1] += 1
            W[w_base+2, rank_offset+1] += 1; W[w_base+3, rank_offset+1] -= 1
            
            # m3 = a00*(b01-b11)
            U[u_base, rank_offset+2] += 1
            V[v_base+1, rank_offset+2] += 1; V[v_base+3, rank_offset+2] -= 1
            W[w_base+1, rank_offset+2] += 1
            
            # m4 = a11*(b10-b00)
            U[u_base+3, rank_offset+3] += 1
            V[v_base+2, rank_offset+3] += 1; V[v_base, rank_offset+3] -= 1
            W[w_base+1, rank_offset+3] += 1; W[w_base+2, rank_offset+3] += 1
            
            # m5 = (a00+a01)*b11
            U[u_base, rank_offset+4] += 1; U[u_base+1, rank_offset+4] += 1
            V[v_base+3, rank_offset+4] += 1
            W[w_base+4, rank_offset+4] -= 1; W[w_base+1, rank_offset+4] += 1
            
            # m6 = (a10-a00)*(b00+b01)
            U[u_base+2, rank_offset+5] += 1; U[u_base, rank_offset+5] -= 1
            V[v_base, rank_offset+5] += 1; V[v_base+1, rank_offset+5] += 1
            W[w_base+3, rank_offset+5] += 1
            
            # m7 = (a01-a11)*(b10+b11)
            U[u_base+1, rank_offset+6] += 1; U[u_base+3, rank_offset+6] -= 1
            V[v_base+2, rank_offset+6] += 1; V[v_base+3, rank_offset+6] += 1
            W[w_base, rank_offset+6] += 1; W[w_base+3, rank_offset+6] -= 1

        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # U indices: 0,1,2,3. V indices: 0,1,2,3. W indices: 0,1,2,3.
        add_strassen(0, 0, 0, 0)
        
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        # U indices: 0,1,2,3. V indices: 4,5,6,7. W indices: 4,5,6,7.
        add_strassen(7, 0, 4, 4)
        
        # Block 3: A[0:2, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2]
        # U indices: 4,5. V indices: 8,9. W indices: 0,1,2,3.
        # Naive 2x1x2
        # C[0,0] = A[0,2] B[2,0]
        U[4, 14] = 1; V[8, 14] = 1; W[0, 14] = 1
        # C[0,1] = A[0,2] B[2,1]
        U[4, 15] = 1; V[9, 15] = 1; W[1, 15] = 1
        # C[1,0] = A[1,2] B[2,0]
        U[5, 16] = 1; V[8, 16] = 1; W[2, 16] = 1
        # C[1,1] = A[1,2] B[2,1]
        U[5, 17] = 1; V[9, 17] = 1; W[3, 17] = 1
        
        # Block 4: A[0:2, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4]
        # U indices: 4,5. V indices: 10,11. W indices: 4,5,6,7.
        # C[0,2] = A[0,2] B[2,2]
        U[4, 18] = 1; V[10, 18] = 1; W[4, 18] = 1
        # C[0,3] = A[0,2] B[2,3]
        U[4, 19] = 1; V[11, 19] = 1; W[5, 19] = 1
        # C[1,2] = A[1,2] B[2,2]
        U[5, 20] = 1; V[10, 20] = 1; W[6, 20] = 1
        # C[1,3] = A[1,2] B[2,3]
        U[5, 21] = 1; V[11, 21] = 1; W[7, 21] = 1
        
        return U, V, W

    def try_rank_19():
        """
        Attempt to find rank 19 by initializing with the 22-term decomposition
        and zeroing out 3 terms, then optimizing.
        """
        U22, V22, W22 = get_22_term_decomp()
        
        # We want to reduce 22 terms to 19.
        # We will zero out 3 of the "Naive" terms (indices 18, 19, 20)
        # and hope the optimizer can absorb their contribution into the other 19.
        # Actually, it's better to zero out 3 random terms or the ones with smallest norm?
        # Let's try zeroing out the last 3 naive terms.
        
        U_init = U22[:, :19].copy()
        V_init = V22[:, :19].copy()
        W_init = W22[:, :19].copy()
        
        # The residual is the sum of the terms we dropped: 19, 20, 21.
        # We are minimizing || T - sum(U_i V_i W_i) ||.
        # The initial sum is sum(T_0..T_18). The target is T.
        # Error = T_19 + T_20 + T_21.
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                lambda x: residual_func(x, 19),
                x0,
                method='lm',
                max_nfev=20000,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            
            if result.cost < 1e-10:
                U_sol = result.x[:nm * 19].reshape(nm, 19)
                V_sol = result.x[nm * 19: nm * 19 + mp * 19].reshape(mp, 19)
                W_sol = result.x[nm * 19 + mp * 19:].reshape(pn, 19)
                
                if verify(U_sol, V_sol, W_sol, tol=1e-6):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
            
        # Try a different set of 3 terms to drop
        # Drop terms 14, 15, 16 (first naive block)
        U_init = np.delete(U22, [14, 15, 16], axis=1)
        V_init = np.delete(V22, [14, 15, 16], axis=1)
        W_init = np.delete(W22, [14, 15, 16], axis=1)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                lambda x: residual_func(x, 19),
                x0,
                method='lm',
                max_nfev=20000,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            
            if result.cost < 1e-10:
                U_sol = result.x[:nm * 19].reshape(nm, 19)
                V_sol = result.x[nm * 19: nm * 19 + mp * 19].reshape(mp, 19)
                W_sol = result.x[nm * 19 + mp * 19:].reshape(pn, 19)
                
                if verify(U_sol, V_sol, W_sol, tol=1e-6):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
            
        return None

    solution = try_rank_19()
    
    if solution is not None:
        U_sol, V_sol, W_sol = solution
        
        def to_list(arr):
            lst = []
            for row in arr:
                row_lst = []
                for val in row:
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-6:
                        row_lst.append(rounded)
                    else:
                        row_lst.append(float(val))
                lst.append(row_lst)
            return lst
        
        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fallback: naive decomposition with rank = n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
