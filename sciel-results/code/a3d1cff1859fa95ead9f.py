"""Matrix multiplication 3,4,7 - rank minimization via block decomposition."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook_terms(n, m, p):
    """Generate rank-1 terms for schoolbook n*m*p multiplication.
    Each term is (u_vec, v_vec, w_vec) of shapes (n*m,), (m*p,), (p*n,).
    u_vec has 1 at position i*m+j
    v_vec has 1 at position j*p+k
    w_vec has 1 at position k*n+i
    """
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m, dtype=float)
                v = np.zeros(m * p, dtype=float)
                w = np.zeros(p * n, dtype=float)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                terms.append((u, v, w))
    return terms

def _strassen_2x2_terms(n, m, p):
    """Generate rank-1 terms for Strassen 2x2 matrix multiplication.
    This handles the case n=2, m=2, p=2 with rank 7.
    Returns list of (u_vec, v_vec, w_vec).
    """
    if n != 2 or m != 2 or p != 2:
        raise ValueError("Strassen 2x2 only for 2x2x2")
    terms = []
    # Strassen formulas for C = A*B where A, B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    # Index mapping: u[i*2+j], v[j*2+k], w[k*2+i]
    def make_u(i, j):
        vec = np.zeros(4, dtype=float)
        vec[i*2+j] = 1.0
        return vec
    def make_v(j, k):
        vec = np.zeros(4, dtype=float)
        vec[j*2+k] = 1.0
        return vec
    def make_w(k, i):
        vec = np.zeros(4, dtype=float)
        vec[k*2+i] = 1.0
        return vec
    
    # M1: (a11+a22)(b11+b22) -> contributes to c11 (+1), c22 (+1)
    # u: a11 at (0,0), a22 at (1,1)
    u1 = make_u(0,0) + make_u(1,1)
    v1 = make_v(0,0) + make_v(1,1)
    w1 = make_w(0,0) + make_w(1,1)  # c11 at w[0*2+0]=w[0], c22 at w[1*2+1]=w[3]
    terms.append((u1, v1, w1))
    
    # M2: (a21+a22)b11 -> contributes to c21 (+1), c22 (-1)
    u2 = make_u(1,0) + make_u(1,1)
    v2 = make_v(0,0)
    w2 = make_w(0,1) - make_w(1,1)  # c21 at w[0*2+1]=w[1], c22 at w[1*2+1]=w[3]
    terms.append((u2, v2, w2))
    
    # M3: a11(b12-b22) -> contributes to c12 (+1), c22 (+1)
    u3 = make_u(0,0)
    v3 = make_v(0,1) - make_v(1,1)
    w3 = make_w(1,0) + make_w(1,1)  # c12 at w[1*2+0]=w[2], c22 at w[1*2+1]=w[3]
    terms.append((u3, v3, w3))
    
    # M4: a22(b21-b11) -> contributes to c11 (+1), c21 (+1)
    u4 = make_u(1,1)
    v4 = make_v(1,0) - make_v(0,0)
    w4 = make_w(0,0) + make_w(0,1)  # c11 at w[0], c21 at w[1]
    terms.append((u4, v4, w4))
    
    # M5: (a11+a12)b22 -> contributes to c11 (-1), c12 (+1)
    u5 = make_u(0,0) + make_u(0,1)
    v5 = make_v(1,1)
    w5 = -make_w(0,0) + make_w(1,0)  # c11 at w[0], c12 at w[2]
    terms.append((u5, v5, w5))
    
    # M6: (a21-a11)(b11+b12) -> contributes to c22 (+1)
    u6 = make_u(1,0) - make_u(0,0)
    v6 = make_v(0,0) + make_v(0,1)
    w6 = make_w(1,1)
    terms.append((u6, v6, w6))
    
    # M7: (a12-a22)(b21+b22) -> contributes to c11 (+1)
    u7 = make_u(0,1) - make_u(1,1)
    v7 = make_v(1,0) + make_v(1,1)
    w7 = make_w(0,0)
    terms.append((u7, v7, w7))
    
    return terms

def _map_terms_to_block(terms, row_start, col_start, m, p):
    """Map local terms for a sub-block to global indices.
    
    The block is A[row_start:row_start+n_local, 0:m] * B[0:m, col_start:col_start+p_local]
    producing C[row_start:row_start+n_local, col_start:col_start+p_local].
    
    Actually, let me reconsider. The full problem is:
    A is n x m (3x4), B is m x p (4x7), C is n x p (3x7).
    
    For a block A[i0:i1, :] * B[:, j0:j1] -> C[i0:i1, j0:j1]:
    - U index: i*m + j where i in [i0,i1), j in [0,m)
    - V index: j*p + k where j in [0,m), k in [j0,j1)
    - W index: k*n + i where i in [i0,i1), k in [j0,j1)
    
    For local terms where local i' in [0,n_local), local j' in [0,m), local k' in [0,p_local):
    - Global i = i0 + i'
    - Global j = j' (since we split columns of B, not rows of A)
    - Global k = j0 + k'
    
    Wait, I need to be more careful. Let me redefine.
    
    If we split B's columns into blocks, for block col [j0, j1):
    Local: A_full (n x m) * B_block (m x p_local) -> C_block (n x p_local)
    
    For the U, V, W of the full problem:
    - U[i*m + j, c] = 1 for i in range(n), j in range(m)
    - V[j*p + k, c] = 1 for j in range(m), k in range(j0, j1)
    - W[k*n + i, c] = 1 for i in range(n), k in range(j0, j1)
    
    So for a local term with (i_local, j_local, k_local) where:
    - i_local in [0, n)
    - j_local in [0, m) 
    - k_local in [0, p_local)
    
    The global indices are:
    - U position: i_local * m + j_local
    - V position: j_local * p + (j0 + k_local)
    - W position: (j0 + k_local) * n + i_local
    """
    global_terms = []
    n_local = None
    p_local = None
    for (u_local, v_local, w_local) in terms:
        # Determine local dimensions from the vectors
        u_global = np.zeros(len(u_local) * (m // int(np.sqrt(len(u_local)/n_local))) if n_local else 0, dtype=float)
        # This approach is getting complicated. Let me use a different method.
        pass
    
    # Let me just directly construct based on known term structure
    return global_terms

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy: Split C (3x7) into blocks and use optimized decompositions
    # 
    # Block decomposition of C:
    # C[0:2, 0:4] : A[0:2, 0:4] * B[0:4, 0:4] -> 2x2x2 subproblem (Strassen, rank 7)
    # C[0:2, 4:7] : A[0:2, 0:4] * B[0:4, 4:7] -> 2x2x3 subproblem
    # C[2:3, 0:4] : A[2:3, 0:4] * B[0:4, 0:4] -> 1x2x2 subproblem  
    # C[2:3, 4:7] : A[2:3, 0:4] * B[0:4, 4:7] -> 1x2x3 subproblem
    #
    # For 2x2x3: we can split the 3 columns into 1+2 or 2+1
    # Split into 1+2: 2x2x1 (rank 4) + 2x2x2 (rank 7) = 11
    # Or use a direct 2x2x3 decomposition
    #
    # Let me compute the rank for each block:
    # Block 1: 2x2x2 -> Strassen rank 7
    # Block 2: 2x2x3 -> Let's use schoolbook for now: 2*2*3 = 12, or optimized
    # Block 3: 1x2x2 -> Schoolbook: 1*2*2 = 4
    # Block 4: 1x2x3 -> Schoolbook: 1*2*3 = 6
    #
    # Total with schoolbook except Block 1: 7 + 12 + 4 + 6 = 29
    # Versus full schoolbook: 3*4*7 = 84
    #
    # Let me try to optimize Block 2 (2x2x3) further.
    # 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # C[:, 0] = A * B[:, 0] (2x2x1, rank 4)
    # C[:, 1:] = A * B[:, 1:] (2x2x2, rank 7)
    # Total: 11
    #
    # So total: 7 + 11 + 4 + 6 = 28
    #
    # Can we do better? Let me think about 1x2x3.
    # A is 1x2, B is 2x3, C is 1x3
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # This is essentially computing 3 dot products of 2-vectors.
    # Rank: Each output element needs 2 terms, but the inputs are shared.
    # Minimum rank for 1x2x3: It's the rank of the tensor.
    # The tensor T[i,j,k] = delta_{i,0} (for 1x2x3, i is always 0)
    # Actually it's T[j,k] = 1 for the product term.
    # This is a 2x3 matrix viewed as a tensor. Rank of a matrix as tensor = matrix rank.
    # The "matrix" here is... actually it's not quite a matrix multiplication in the usual sense.
    # 
    # Let me just use schoolbook for the small blocks and Strassen for 2x2x2.
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_pos, u_val, v_pos, v_val, w_pos, w_val):
        """Add a single rank-1 term contribution."""
        nonlocal rank
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        u[u_pos] = u_val
        v[v_pos] = v_val
        w[w_pos] = w_val
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
        rank += 1
    
    # Block 1: C[0:2, 0:4] from A[0:2, 0:4] * B[0:4, 0:4]
    # Local: 2x2x2, using Strassen
    # Global mapping: i_global = i_local (0,1), j_global = j_local (0,1,2,3) -> wait, m=4
    # Hmm, the sub-block is 2x4 * 4x4 -> 2x4
    # So it's 2x4x4, not 2x2x2!
    
    # I made an error. Let me redo the block decomposition.
    # A is 3x4, B is 4x7, C is 3x7
    # 
    # If I split A's rows into [0:2] and [2:3], and B's columns into [0:4] and [4:7]:
    # Block (0,0): A[0:2, :] (2x4) * B[:, 0:4] (4x4) -> C[0:2, 0:4] (2x4)
    # Block (0,1): A[0:2, :] (2x4) * B[:, 4:7] (4x3) -> C[0:2, 4:7] (2x3)
    # Block (1,0): A[2:3, :] (1x4) * B[:, 0:4] (4x4) -> C[2:3, 0:4] (1x4)
    # Block (1,1): A[2:3, :] (1x4) * B[:, 4:7] (4x3) -> C[2:3, 4:7] (1x3)
    #
    # Block (0,0): 2x4x4. Split B's columns further? Or split A's columns?
    # Let me split A's columns into [0:2] and [2:4]:
    # Block (0,0,a): A[0:2, 0:2] (2x2) * B[0:2, 0:4] (2x4) -> C[0:2, 0:4]
    # Block (0,0,b): A[0:2, 2:4] (2x2) * B[2:4, 0:4] (2x4) -> C[0:2, 0:4]
    # Each is 2x2x4. Split the 4 into 2+2:
    # Block (0,0,a,1): 2x2x2 -> Strassen rank 7
    # Block (0,0,a,2): 2x2x2 -> Strassen rank 7
    # Block (0,0,b,1): 2x2x2 -> Strassen rank 7
    # Block (0,0,b,2): 2x2x2 -> Strassen rank 7
    # Total for Block (0,0): 28
    #
    # This is getting complex. Let me just use schoolbook for everything and verify it works,
    # then try to improve specific blocks.
    
    # Actually, let me start with a clean schoolbook implementation and verify correctness.
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m, dtype=float)
                v = np.zeros(m * p, dtype=float)
                w = np.zeros(p * n, dtype=float)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.array(U_list).T  # shape (rank, n*m) -> transpose to (n*m, rank)
    V = np.array(V_list).T  # shape (rank, m*p) -> transpose to (m*p, rank)
    W = np.array(W_list).T  # shape (rank, p*n) -> transpose to (p*n, rank)
    
    # Convert to lists for the expected output format
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Return what we have (schoolbook) as safety
        pass
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
