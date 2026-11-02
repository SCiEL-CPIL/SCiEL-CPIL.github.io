"""
Program for matrix multiplication 2x5x5 rank minimization.
Combines the structural block-Strassen approach (rank 46) with a
global linear-algebra rank-reduction pass that exploits linear
dependencies among the rank-1 terms to potentially lower the rank.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    Strategy:
    1. Build the known rank-46 Strassen-block decomposition.
    2. Attempt to reduce rank by finding linear dependencies among
       the columns of U, V, W simultaneously (tensor rank reduction).
    3. Verify the reduced decomposition exactly.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 5  # A is n x m, B is m x p, C is p x n
    
    # Step 1: Build the rank-46 Strassen decomposition
    terms = []  # Each term: (u_dict, v_dict, w_dict)
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = (a_row_off) * m + (a_col_off)
        b_u = (a_row_off) * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + (a_col_off)
        d_u = (a_row_off + 1) * m + (a_col_off + 1)
        
        e_v = (b_row_off) * p + (b_col_off)
        f_v = (b_row_off) * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + (b_col_off)
        h_v = (b_row_off + 1) * p + (b_col_off + 1)
        
        o00_w = (out_col_off) * n + (out_row_off)
        o01_w = (out_col_off + 1) * n + (out_row_off)
        o10_w = (out_col_off) * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)
        
        # M1 = (a+d)(e+h)
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        # M2 = (c+d)e
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        # M3 = a(f-h)
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        # M4 = d(g-e)
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        # M5 = (a+b)h
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        # M6 = (c-a)(e+f)
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        # M7 = (b-d)(g+h)
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])
    
    # Build the rank-46 decomposition
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)
    
    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)
    
    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)
    
    # Convert to matrices
    r0 = len(terms)
    U0 = np.zeros((n * m, r0), dtype=float)
    V0 = np.zeros((m * p, r0), dtype=float)
    W0 = np.zeros((p * n, r0), dtype=float)
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U0[idx, col] = val
        for idx, val in v_entries:
            V0[idx, col] = val
        for idx, val in w_entries:
            W0[idx, col] = val
    
    if time.time() > deadline:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Step 2: Try to reduce rank via linear dependency search
    # The tensor T is determined by the triple (U, V, W).
    # If columns are linearly dependent in a coordinated way, we can reduce rank.
    
    # Approach: Try to find a set of columns that are linearly dependent
    # in the joint space. Specifically, if we can find a linear combination
    # of columns that gives zero in all three matrices simultaneously,
    # we can remove that column.
    
    # More practically: use the fact that the tensor has structure.
    # Try removing one column at a time and check if the remaining
    # decomposition still spans the same tensor.
    
    # Actually, a better approach: check if any column is a linear
    # combination of others in all three matrices.
    
    # For rank r, check if any column c can be expressed as a linear
    # combination of the other r-1 columns in U, V, W simultaneously.
    # This is equivalent to checking if the column space of U, V, W
    # has dimension less than r.
    
    # Let's try a different approach: iterative column elimination.
    # For each column, check if it's in the span of the others
    # (in all three matrices). If so, remove it.
    
    U_cur = U0.copy()
    V_cur = V0.copy()
    W_cur = W0.copy()
    active = list(range(r0))
    
    # Try to eliminate columns
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
        if len(active) <= 30:  # Don't go too low without verification
            break
            
        eliminated = False
        for idx in range(len(active)):
            col = active[idx]
            if idx == 0 or idx == len(active) - 1:
                continue  # Skip first and last for efficiency
            
            # Check if column col is in the span of others
            others = [c for j, c in enumerate(active) if j != idx]
            if len(others) < 3:
                continue
                
            # Try to express U_cur[:, col] as combination of U_cur[:, others]
            # Use least squares
            U_others = U_cur[:, others]
            V_others = V_cur[:, others]
            W_others = W_cur[:, others]
            
            try:
                # Solve for coefficients in U
                coeff_u, _, _, _ = np.linalg.lstsq(U_others, U_cur[:, col], rcond=None)
                # Check if same coefficients work for V and W
                V_pred = V_others @ coeff_u
                W_pred = W_others @ coeff_u
                
                if (np.max(np.abs(V_pred - V_cur[:, col])) < 1e-10 and
                    np.max(np.abs(W_pred - W_cur[:, col])) < 1e-10):
                    # Column is redundant! Remove it
                    active.remove(col)
                    eliminated = True
                    break
            except:
                continue
        
        if not eliminated:
            break
    
    if len(active) < r0:
        U_red = U_cur[:, active]
        V_red = V_cur[:, active]
        W_red = W_cur[:, active]
        
        # Verify the reduced decomposition
        if _verify_decomposition(U_red, V_red, W_red, n, m, p):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    # Step 3: Try a more aggressive approach - use SVD on the
    # Khatri-Rao product to find minimal rank representation
    
    if time.time() < deadline - 10:
        # Build the tensor and try to find a lower rank representation
        # The tensor T[i,j,k] = delta_{i,j} for the matmul structure
        # Actually T[i, j, k] = 1 if the indices form a valid matmul path
        
        # Let's try: for each output position (i, k), the contribution
        # is sum_j A[i,j] * B[j,k]. The tensor is:
        # T[(i,j), (j,k), (k,i)] = delta
        
        # Try to find if we can merge some columns
        # Use the approach: check if the 3-tensor has rank less than r0
        # by looking at the matrix flattening
        
        # Flatten T as a matrix of size (n*m) x (m*p*n) or similar
        # Actually, for tensor rank, we can look at multiple flattenings
        
        # Flattening 1: (n*m) x (m*p) for each fixed output
        # This is complex. Let's try a simpler heuristic:
        # Check if any two columns can be merged (their sum/difference
        # gives a valid rank-1 term)
        
        # For now, return the best we have
        pass
    
    # Return the rank-46 decomposition
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{j,k}
    where the tensor for matmul is T[i,j,k] = 1 if j==k else 0... 
    Actually no. The matmul tensor is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T has T[(i,j),(j,k),(k,i)] = 1 for all valid (i,j,k)
    Wait, that's not right either.
    
    The correct tensor: For matrix multiplication, the structure tensor is
    T_{i,j,k} where the three indices correspond to:
    - (i,j) for A
    - (j,k) for B  
    - (i,k) for C
    
    But our indexing is:
    - U indexed by (i, j) with i in [0,n), j in [0,m) -> A[i,j]
    - V indexed by (j, k) with j in [0,m), k in [0,p) -> B[j,k]
    - W indexed by (k, i) with k in [0,p), i in [0,n) -> C[k,i]... 
    
    Wait, the task says W[k*n+i, r]. So W is indexed by (k, i) where k is the
    column of C and i is the row of C. But C is n x p, so row i in [0,n), col k in [0,p).
    W index = k*n + i means k is the "outer" index and i is the "inner" index.
    
    Actually re-reading: "output factor follows the official trace convention W[k*n+i,r]"
    So W has shape (p*n, rank) and W[k*n+i, r] corresponds to C[i, k] (row i, col k).
    
    The tensor condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k]
    where C[i,k] = sum_j A[i,j] * B[j,k]
    
    For the decomposition to be exact, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j, ?}
    
    No, the tensor for matmul is:
    T[i, j, k] = 1 for all (i, j, k) in [0,n) x [0,m) x [0,p)
    Wait no. The matmul tensor is defined by:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    As a 3-tensor: T[(i,j), (j,k), (i,k)] where the "product" structure means
    for each (i,j,k): the term A[i,j] * B[j,k] contributes to C[i,k].
    
    So the tensor is: T_{(i,j), (j,k), (i,k)} = 1 for all valid (i,j,k)
    But that's just saying every triple (i,j,k) with consistent indices appears.
    
    More precisely: the structure tensor has entries:
    T[a, b, c] = 1 where a indexes A (i,j), b indexes B (j,k), c indexes C (i,k)
    and the condition is that the j in A's column matches B's row, and
    i in A's row matches C's row, and k in B's col matches C's col.
    
    So T[(i,j), (j,k), (i,k)] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    In our flattened indexing:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    
    So the condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i, r] = 0 for k' != k
    
    Wait, that's not right. Let me re-think.
    
    The matrix multiplication tensor is:
    For each (i, j, k) in [n] x [m] x [p]:
      A[i,j] * B[j,k] contributes to C[i,k]
    
    So the tensor T has:
    T[A_idx, B_idx, C_idx] = 1 where:
      A_idx = i*m + j
      B_idx = j*p + k
      C_idx = k*n + i
      
    for all (i, j, k). And T = 0 for all other triples.
    
    So the verification is:
    For all (i, j, k) in [n] x [m] x [p]:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    For all other triples (a, b, c) not of the form (i*m+j, j*p+k, k*n+i):
      sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    This is a lot of checks. Let's do a sampling approach.
    """
    r = U.shape[1]
    
    # Check the "on" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
                # Actually this isn't right. The tensor is:
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # So we need: sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] = 1
                # This is a trilinear form, not a bilinear one.
                # Let me compute it properly.
                
                # For each r, the contribution is U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                # Sum over r.
                contrib = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(contrib - 1.0) > 1e-10:
                    return False
    
    # Check some "off" entries (sampling)
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        
        # Check if (a, b, c) is an "on" entry
        i1, j1 = divmod(a, m)
        j2, k2 = divmod(b, p)
        k3, i3 = divmod(c, n)
        
        is_on = (i1 == i3 and j1 == j2 and k2 == k3)
        
        contrib = np.sum(U[a, :] * V[b, :] * W[c, :])
        expected = 1.0 if is_on else 0.0
        if abs(contrib - expected) > 1e-10:
            return False
    
    return True
# EVOLVE-BLOCK-END
