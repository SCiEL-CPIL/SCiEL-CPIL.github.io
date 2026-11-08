"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Build terms: list of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    terms = []
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        terms.append((u_indices, u_vals, v_indices, v_vals, w_indices, w_vals))
    
    # Split: rows [0]|[1,2], middle [0,1]|[2,3,4], cols [0,1]|[2,3]|[4,5]
    for cb in range(3):
        col_start = cb * 2
        
        # Row 0, middle[0:2], cols[col_start:col_start+2]: 1x2x2, rank 4
        for a in range(2):
            for b in range(2):
                add_term([0*5+a], [1], [a*6+col_start+b], [1], [(col_start+b)*3+0], [1])
        
        # Row 0, middle[2:5], cols[col_start:col_start+2]: 1x3x2, rank 6
        for a in range(2, 5):
            for b in range(2):
                add_term([0*5+a], [1], [a*6+col_start+b], [1], [(col_start+b)*3+0], [1])
        
        # Rows [1,2], middle[0:2], cols[col_start:col_start+2]: 2x2x2, Strassen rank 7
        a11_u, a12_u = 1*5+0, 1*5+1
        a21_u, a22_u = 2*5+0, 2*5+1
        b11_v, b12_v = 0*6+col_start, 0*6+col_start+1
        b21_v, b22_v = 1*6+col_start, 1*6+col_start+1
        c11_w, c12_w = col_start*3+1, col_start*3+2
        c21_w, c22_w = col_start*3+3, col_start*3+4
        
        add_term([a11_u, a22_u], [1,1], [b11_v, b22_v], [1,1], [c11_w, c22_w], [1,1])
        add_term([a21_u, a22_u], [1,1], [b11_v], [1], [c21_w, c22_w], [1,-1])
        add_term([a11_u], [1], [b21_v, b22_v], [1,-1], [c12_w, c22_w], [1,1])
        add_term([a22_u], [1], [b12_v, b21_v], [1,-1], [c11_w, c21_w], [1,1])
        add_term([a11_u, a12_u], [1,1], [b22_v], [1], [c11_w, c12_w], [-1,1])
        add_term([a21_u, a11_u], [1,-1], [b11_v, b12_v], [1,1], [c22_w], [1])
        add_term([a12_u, a22_u], [1,-1], [b21_v, b22_v], [1,1], [c11_w], [1])
        
        # Rows [1,2], middle[2:5], cols[col_start:col_start+2]: 2x3x2, rank 12
        for a in range(2, 5):
            for b in range(2):
                for r in range(1, 3):
                    add_term([r*5+a], [1], [a*6+col_start+b], [1], [(col_start+b)*3+r], [1])
    
    rank = len(terms)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for t, (ui, uv, vi, vv, wi, wv) in enumerate(terms):
        for i, v in zip(ui, uv):
            U[i][t] = v
        for i, v in zip(vi, vv):
            V[i][t] = v
        for i, v in zip(wi, wv):
            W[i][t] = v
    
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Our decomposition: C = sum_t U[:,t] * V[:,t] * W[:,t]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, t] is the coefficient for C[i,k]
    
    C_computed = np.zeros((n, p))
    for t in range(rank):
        u_col = U_np[:, t].reshape(n, m)
        v_col = V_np[:, t].reshape(m, p)
        w_col = W_np[:, t].reshape(p, n)
        # Contribution to C[i,k] = u_col[i,j] * v_col[j,k] * w_col[k,i] summed over j
        # Actually: sum_j u_col[i,j] * v_col[j,k] * w_col[k,i]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += np.sum(u_col[i, :] * v_col[:, k]) * w_col[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
