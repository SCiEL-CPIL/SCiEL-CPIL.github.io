import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    dim_u = n * m  # 8
    dim_v = m * p  # 20
    dim_w = p * n  # 10
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def build_tensor():
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[u_idx(i, j), v_idx(j, k), w_idx(k, i)] = 1.0
        return T
    
    T = build_tensor()
    
    def verify_fast(U, V, W, rank):
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_decomp, atol=1e-10)
    
    # Trivial rank 40 decomposition
    def trivial_decomp():
        rank_t = 40
        U_t = np.zeros((dim_u, rank_t))
        V_t = np.zeros((dim_v, rank_t))
        W_t = np.zeros((dim_w, rank_t))
        col_t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[u_idx(i,j), col_t] = 1.0
                    V_t[v_idx(j,k), col_t] = 1.0
                    W_t[w_idx(k,i), col_t] = 1.0
                    col_t += 1
        return U_t, V_t, W_t, rank_t
    
    # Block Strassen-based rank 36 decomposition
    def strassen_decomp():
        rank = 36
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            # M1: (y00 + y11) * (z00 + z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M2: (y10 + y11) * z00
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            # M3: y00 * (z01 - z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M4: y11 * (z10 - z00)
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            # M5: (y00 + y01) * z11
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            # M6: (y10 - y00) * (z00 + z01)
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M7: (y01 - y11) * (z10 + z11)
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
        # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
        # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
        # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[2:4,0:2]
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
        
        # Block 5: A[0:2,0:2] * B[0:2,4] -> C[4,0:2] (2x2 * 2x1, rank 4)
        U[u_idx(0,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        
        # Block 6: A[0:2,2:4] * B[2:4,4] -> C[4,0:2] (2x2 * 2x1, rank 4)
        U[u_idx(0,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        
        assert col == rank, f"Expected rank {rank}, got {col}"
        
        if not verify_fast(U, V, W, rank):
            return None
        
        return U, V, W, rank
    
    # Try to reduce rank by removing terms
    def try_reduce_rank(U, V, W, rank, target_rank):
        if rank <= target_rank:
            return None
        
        for r_remove in range(rank):
            if time.time() > deadline:
                break
            
            U_new = np.delete(U, r_remove, axis=1)
            V_new = np.delete(V, r_remove, axis=1)
            W_new = np.delete(W, r_remove, axis=1)
            
            if verify_fast(U_new, V_new, W_new, rank - 1):
                return U_new, V_new, W_new, rank - 1
        
        return None
    
    # Build rank-36 Strassen-based decomposition
    result = strassen_decomp()
    if result is None:
        U, V, W, rank = trivial_decomp()
    else:
        U, V, W, rank = result
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find rank 35 via structured search
    if time.time() < deadline:
        result = try_reduce_rank(U, V, W, rank, 35)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
    
    # Try more aggressive rank reduction
    if time.time() < deadline and best_rank > 34:
        # Try removing multiple terms
        for r1 in range(best_rank):
            for r2 in range(r1 + 1, best_rank):
                if time.time() > deadline:
                    break
                
                U_new = np.delete(best_U, [r1, r2], axis=1)
                V_new = np.delete(best_V, [r1, r2], axis=1)
                W_new = np.delete(best_W, [r1, r2], axis=1)
                
                if verify_fast(U_new, V_new, W_new, best_rank - 2):
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, best_rank - 2
                    break
            if time.time() > deadline:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
