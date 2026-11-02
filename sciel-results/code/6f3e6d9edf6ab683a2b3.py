"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def build_tensor():
        """Build the multiplication tensor of shape (n*m, m*p, p*n)."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return T
    
    T = build_tensor()
    
    def build_strassen_36():
        """Build the rank-36 block-Strassen decomposition."""
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            U[u_idx(y00[0], y00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = -1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = -1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y01[0], y01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y00[0], y00[1])][col] = -1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            U[u_idx(y01[0], y01[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = -1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
        
        # 2x2x1 blocks for column 4
        for (yi,yj), (zj,zk), (wk,wi) in [
            ((0,0),(0,4),(4,0)), ((0,1),(1,4),(4,0)),
            ((1,0),(0,4),(4,1)), ((1,1),(1,4),(4,1)),
            ((0,2),(2,4),(4,0)), ((0,3),(3,4),(4,0)),
            ((1,2),(2,4),(4,1)), ((1,3),(3,4),(4,1)),
        ]:
            U[u_idx(yi,yj)][col] = 1
            V[v_idx(zj,zk)][col] = 1
            W[w_idx(wk,wi)][col] = 1
            col += 1
        
        return U, V, W
    
    U, V, W = build_strassen_36()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)
    
    # Verify initial decomposition
    if not verify(U, V, W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    def can_remove_columns(U, V, W, cols_to_remove):
        """Check if removing the given columns still gives a valid decomposition."""
        if time.time() > deadline - 5:
            return False
        mask = np.ones(U.shape[1], dtype=bool)
        mask[list(cols_to_remove)] = False
        U_rem = U[:, mask]
        V_rem = V[:, mask]
        W_rem = W[:, mask]
        if U_rem.shape[1] == 0:
            return False
        T_rem = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem)
        return np.allclose(T_rem, T, atol=1e-9)
    
    # Iteratively try to remove single columns
    while time.time() < deadline - 20:
        removed = False
        for c in range(U.shape[1]):
            if time.time() > deadline - 20:
                break
            if can_remove_columns(U, V, W, [c]):
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = U.shape[1]
                removed = True
                break
        if not removed:
            break
    
    # Try pair removals
    if time.time() < deadline - 15:
        r = U.shape[1]
        for c1 in range(r):
            if time.time() > deadline - 15:
                break
            for c2 in range(c1+1, r):
                if time.time() > deadline - 15:
                    break
                if can_remove_columns(U, V, W, [c1, c2]):
                    mask = np.ones(r, dtype=bool)
                    mask[[c1,c2]] = False
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = U.shape[1]
                    break
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
