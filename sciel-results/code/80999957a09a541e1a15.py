import time
import numpy as np
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build rank 28 solution as baseline
    def build_rank28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: -1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        def add_term(u_dict, v_dict, w_dict, col):
            for idx, coeff in u_dict.items():
                U[idx, col] += coeff
            for idx, coeff in v_dict.items():
                V[idx, col] += coeff
            for idx, coeff in w_dict.items():
                W[idx, col] += coeff
        
        def apply_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen):
                col = start_col + t_idx
                u_g = {}
                for loc, coeff in u_loc.items():
                    r, c = divmod(loc, 2)
                    g_idx = r * m + (a_col_off * 2 + c)
                    u_g[g_idx] = u_g.get(g_idx, 0) + coeff
                v_g = {}
                for loc, coeff in v_loc.items():
                    r, c = divmod(loc, 2)
                    g_idx = (b_row_off * 2 + r) * p + (b_col_off * 2 + c)
                    v_g[g_idx] = v_g.get(g_idx, 0) + coeff
                w_g = {}
                for loc, coeff in w_loc.items():
                    r, c = divmod(loc, 2)
                    g_idx = (c_col_off * 2 + c) * n + r
                    w_g[g_idx] = w_g.get(g_idx, 0) + coeff
                add_term(u_g, v_g, w_g, col)
        
        apply_block(0, 0, 0, 0, 0)
        apply_block(0, 0, 1, 1, 7)
        apply_block(1, 1, 0, 0, 14)
        apply_block(1, 1, 1, 1, 21)
        
        return U, V, W
    
    U28, V28, W28 = build_rank28()
    
    def verify(U, V, W, seed=42):
        np.random.seed(seed)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-10)
    
    # Try to find rank 20 by solving the tensor decomposition
    # We need: for all i in [0,2), j in [0,4), k in [0,4):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is 32 equations.
    # Variables: U (8*rank), V (16*rank), W (8*rank) = 32*rank variables
    # For rank 20: 640 variables, 32 equations. Underdetermined.
    
    # Strategy: Fix W to have simple structure, then solve for U and V
    # Or use iterative approach
    
    # Let's try: set W[k*n+i, r] = 1 for all k,i,r (rank 20)
    # Then we need: sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all i,j,k
    # This means U @ V^T (with appropriate indexing) should give identity-like structure
    # Actually, for fixed i,k: sum_r U[i*m+j,r] * V[j*p+k,r] summed over j? No.
    # The equation is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # If W[k*n+i,r] = 1, then sum_r U[i*m+j,r] * V[j*p+k,r] = 1 for all i,j,k
    # But U[i*m+j,r] doesn't depend on k, and V[j*p+k,r] doesn't depend on i.
    # So for fixed j: sum_r U[i*m+j,r] * V[j*p+k,r] must be 1 for all i,k
    # This means for each j, the vector [U[i*m+j,r] for all i,r] dotted with [V[j*p+k,r] for all k,r] = 1 for all i,k
    # This is very restrictive.
    
    # Better: use the structure. Let's try rank 20 with a specific pattern.
    # Given the difficulty, return rank 28.
    
    if not verify(U28, V28, W28):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("FAILED")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
