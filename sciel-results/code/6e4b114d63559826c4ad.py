import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0
    
    # Build the rank-68 decomposition:
    # - 4 Strassen blocks: A[0:2,0:2] * B[0:2, c:c+2] for c in {0,2,4,6} -> 4*7 = 28 terms
    # - Direct: A[0:2,2] * B[2,0:8] -> 2*8 = 16 terms
    # - Direct: A[2,0:2] * B[0:2,0:8] -> 2*8 = 16 terms
    # - Direct: A[2,2] * B[2,0:8] -> 8 terms
    # Total: 28 + 16 + 16 + 8 = 68
    
    U_t = []  # list of lists of ((i,j), coeff)
    V_t = []  # list of lists of ((j,k), coeff)
    W_t = []  # list of lists of ((i,k), coeff)
    
    def strassen_decomp(a, b, c, d, e, f, g, h, C00, C01, C10, C11):
        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    # 4 Strassen blocks for A[0:2,0:2] * B[0:2, c:c+2]
    for col_off in range(0, 8, 2):
        # A sub-block indices: (0,0), (0,1), (1,0), (1,1)
        # B sub-block indices: (0,col_off), (0,col_off+1), (1,col_off), (1,col_off+1)
        # C output: (0,col_off), (0,col_off+1), (1,col_off), (1,col_off+1)
        a = (0, 0); b = (0, 1); c = (1, 0); d = (1, 1)
        e = (0, col_off); f = (0, col_off + 1); g = (1, col_off); h = (1, col_off + 1)
        C00 = (0, col_off); C01 = (0, col_off + 1); C10 = (1, col_off); C11 = (1, col_off + 1)
        strassen_decomp(a, b, c, d, e, f, g, h, C00, C01, C10, C11)
    
    # Direct: A[0:2,2] * B[2,0:8]
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k)
    
    # Direct: A[2,0:2] * B[0:2,0:8]
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k)
    
    # Direct: A[2,2] * B[2,0:8]
    for k in range(8):
        direct_term(2, 2, k)
    
    rank = len(W_t)
    
    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff
    
    # Verify correctness
    rng = np.random.RandomState(42)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        C_computed += np.outer(np.zeros((n, m)), np.zeros((m, p)))  # placeholder
        # More efficiently:
        # C_computed[i,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Efficient verification using einsum-like approach
    C_computed = np.zeros((n, p))
    for r in range(rank):
        # For this r, compute contribution
        u_col = U[:, r]  # shape (n*m,)
        v_col = V[:, r]  # shape (m*p,)
        w_col = W[:, r]  # shape (p*n,)
        
        # Reshape u_col to (n, m) and v_col to (m, p)
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        # Product gives (n, p)
        prod = u_mat @ v_mat  # (n, p)
        # Now multiply element-wise by w_col reshaped to (p, n) transposed
        w_mat = w_col.reshape(p, n)  # W is indexed W[k*n+i] so row k, col i
        C_computed += prod * w_mat.T
    
    if np.max(np.abs(C_computed - C_expected)) > 1e-9:
        # Fallback: naive rank-72
        U_t2 = []
        V_t2 = []
        W_t2 = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    direct_term(i, j, k)
                    # Oops, I was appending to the original lists. Let me fix this.
        
        # Actually the direct_term above appends to U_t, V_t, W_t which already have 68 entries.
        # Let me just do the fallback properly.
        U_fb = np.zeros((n * m, n * m * p))
        V_fb = np.zeros((m * p, n * m * p))
        W_fb = np.zeros((p * n, n * m * p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
                    idx += 1
        U, V, W = U_fb, V_fb, W_fb
        rank = n * m * p
    
    # Convert to nice lists
    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v * 2) / 2) < 1e-9:
                    nv = round(v * 2) / 2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res
    
    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
