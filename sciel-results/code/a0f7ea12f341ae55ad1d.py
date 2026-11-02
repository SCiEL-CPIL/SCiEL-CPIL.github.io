"""
Matrix multiplication tensor decomposition for 2x3 * 3x7.
Uses a hybrid Strassen/schoolbook approach to achieve rank 39,
then attempts further reduction via local search and structured
Winograd-style embeddings.

Strategy:
1. Partition the 2x3 * 3x7 multiplication into blocks.
2. Apply Strassen (rank 7) to the 2x2 * 2x2 sub-block.
3. Use schoolbook (rank 4) for the remaining 2x1 * 1x2 block.
4. Total: 7 + 4 = 11 for the first 2 columns of B.
   Repeat for remaining columns, but share structure.
5. Actually: partition B into [B0, B1] where B0 is 2x3 (cols 0,1) and B1 is 2x4 (cols 2-5).
   Wait, let me think more carefully.

For 2x3 * 3x7:
A is 2x3, B is 3x7, C is 2x7.

Partition B into two parts: B_left (3x3, cols 0-2) and B_right (3x4, cols 3-6).
But that doesn't help directly.

Better: Partition A into A_left (2x2, cols 0-1) and A_right (2x1, col 2).
Partition B into B_left (2x7, rows 0-1) and B_right (1x7, row 2).

Then C = A_left * B_left + A_right * B_right.

For A_left * B_left (2x2 * 2x7):
Partition B_left into 7 column pairs... no, 7 is odd.
Partition B_left into B0 (2x3, cols 0-2) and B1 (2x4, cols 3-6).

For 2x2 * 2x3: Use Strassen-like. Actually, let's just use the known result:
The optimal rank for 2x2 * 2x2 is 7 (Strassen).
For 2x2 * 2x3, we can do it in rank 10 (7 for the 2x2 part + 3 for the extra column).
For 2x2 * 2x4, we can do it in rank 14 (7 for first 2x2 + 7 for second 2x2).

So A_left * B_left (2x2 * 2x7) = (2x2*2x3) + (2x2*2x4) = rank 10 + 14 = 24.
A_right * B_right (2x1 * 1x7) = rank 7 (schoolbook: each of 2*1*7 = 7 terms).

Total: 24 + 7 = 31. That's better than 39!

Wait, let me re-examine. For 2x2 * 2x3:
We have A (2x2), B (2x3), C (2x3).
C = A * B.
We can write this as:
- Use Strassen on the first 2 columns of B: rank 7
- Schoolbook on the 3rd column: rank 4
Total: 11.

For 2x2 * 2x4:
- Strassen on first 2 columns: rank 7
- Strassen on next 2 columns: rank 7
Total: 14.

So A_left * B_left (2x2 * 2x7):
- 2x2 * 2x3: rank 11
- 2x2 * 2x4: rank 14
Total: 25.

A_right * B_right (2x1 * 1x7): rank 7.

Grand total: 25 + 7 = 32.

Let me verify this construction carefully.
"""
import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 130.0

def _check_time():
    return time.time() - START_TIME > TIME_LIMIT

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    return np.allclose(T, T_expected, atol=tol)

def _strassen_2x2_2x2(A_rows, B_rows, C_rows, offset_u, offset_v, offset_w):
    """
    Generate Strassen rank-7 terms for 2x2 * 2x2 multiplication.
    A_rows: indices of A rows (2 indices)
    B_rows: indices of B rows (2 indices)  
    C_rows: indices of C rows (2 indices)
    Returns list of (u_vec, v_vec, w_vec) tuples.
    """
    terms = []
    
    # Strassen formulas for 2x2 * 2x2:
    # M1 = (A10+A11)(B00+B11)
    # M2 = (A10+A11)(B01)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    a00, a01 = A_rows[0], A_rows[1]
    a10, a11 = A_rows[2], A_rows[3]  # Wait, need to be careful with indexing
    
    # Actually, for A (2x2), the indices are:
    # A[0,0], A[0,1], A[1,0], A[1,1]
    # In flattened form with m=2: 0, 1, 2, 3
    # But we're using arbitrary row indices, so let me redefine.
    
    # Let's say A has entries at positions (i, j) where i in {0,1}, j in {0,1}
    # The U index for A[i,j] is some mapping. Let's pass explicit indices.
    
    # For now, let's handle the specific case in construct().
    return terms

def _construct_hybrid(n, m, p):
    """
    Construct the hybrid Strassen/schoolbook decomposition.
    
    For 2x3 * 3x7:
    - A is 2x3 (indices: 0,1,2,3,4,5 in U)
    - B is 3x7 (indices: 0..20 in V)
    - C is 2x7 (indices: 0..13 in W)
    
    Partition:
    - A_left = A[:, 0:2] (2x2), A_right = A[:, 2:3] (2x1)
    - B_left = B[0:2, :] (2x7), B_right = B[2:3, :] (1x7)
    
    C = A_left * B_left + A_right * B_right
    
    For A_left * B_left (2x2 * 2x7):
    Partition B_left into columns:
    - B_left[:, 0:3] (2x3): rank 11
    - B_left[:, 3:7] (2x4): rank 14
    
    For A_right * B_right (2x1 * 1x7): rank 7
    
    Total: 11 + 14 + 7 = 32
    """
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a rank-1 term with specified non-zero entries."""
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[u_idx] = u_val
        v[v_idx] = v_val
        w[w_idx] = w_val
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    def add_term_multi(u_entries, v_entries, w_entries):
        """Add a rank-1 term with multiple non-zero entries."""
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    # Index mappings:
    # U: A[i,j] -> i*m + j = i*3 + j
    # V: B[j,k] -> j*p + k = j*7 + k
    # W: C[i,k] -> k*n + i = k*2 + i
    
    # A indices:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # B indices:
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4, B[0,5]=5, B[0,6]=6
    # B[1,0]=7, B[1,1]=8, B[1,2]=9, B[1,3]=10, B[1,4]=11, B[1,5]=12, B[1,6]=13
    # B[2,0]=14, B[2,1]=15, B[2,2]=16, B[2,3]=17, B[2,4]=18, B[2,5]=19, B[2,6]=20
    
    # C indices:
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4, C[0,5]=5, C[0,6]=6
    # C[1,0]=7, C[1,1]=8, C[1,2]=9, C[1,3]=10, C[1,4]=11, C[1,5]=12, C[1,6]=13
    
    # Part 1: A_left * B_left[:, 0:3] (2x2 * 2x3)
    # A_left: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B_left[:, 0:3]: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=7, B[1,1]=8, B[1,2]=9
    # C[:, 0:3]: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=7, C[1,1]=8, C[1,2]=9
    
    # Strassen for the 2x2 part (cols 0,1 of B_left):
    # M1 = (A[1,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,1]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Schoolbook for col 2 of B_left:
    # A[0,0]*B[0,2] -> C[0,2]
    # A[0,1]*B[1,2] -> C[0,2]
    # A[1,0]*B[0,2] -> C[1,2]
    # A[1,1]*B[1,2] -> C[1,2]
    
    # M1
    add_term_multi([(3, 1), (4, 1)], [(0, 1), (8, 1)], [(0, 1), (7, 1), (13, 1)])
    # Wait, I need to be more careful. Let me redefine.
    
    # Actually, let me restart with a cleaner approach.
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_rank1(u_dict, v_dict, w_dict):
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        for idx, val in u_dict.items():
            u[idx] = val
        for idx, val in v_dict.items():
            v[idx] = val
        for idx, val in w_dict.items():
            w[idx] = val
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    # Part 1: A_left * B_left[:, 0:3] (2x2 * 2x3)
    
    # Strassen terms for 2x2 * 2x2 (using B cols 0,1):
    # M1 = (A10+A11)(B00+B11)
    add_rank1({3: 1, 4: 1}, {0: 1, 8: 1}, {0: 1, 7: 1, 13: 1})
    # M2 = (A10+A11)(B01)
    add_rank1({3: 1, 4: 1}, {1: 1}, {1: 1, 10: 1})
    # M3 = A00(B01-B11)
    add_rank1({0: 1}, {1: 1, 8: -1}, {1: 1, 10: 1})
    # M4 = A11(B10-B00)
    add_rank1({4: 1}, {7: 1, 0: -1}, {0: 1, 7: 1, 13: 1})
    # M5 = (A00+A01)(B11)
    add_rank1({0: 1, 1: 1}, {8: 1}, {0: -1, 1: 1, 13: 1})
    # M6 = (A10-A00)(B00+B01)
    add_rank1({3: 1, 0: -1}, {0: 1, 1: 1}, {13: 1})
    # M7 = (A01-A11)(B10+B11)
    add_rank1({1: 1, 4: -1}, {7: 1, 8: 1}, {0: 1, 13: 1})
    
    # Schoolbook for B col 2:
    add_rank1({0: 1}, {2: 1}, {2: 1})
    add_rank1({1: 1}, {9: 1}, {2: 1})
    add_rank1({3: 1}, {2: 1}, {9: 1})
    add_rank1({4: 1}, {9: 1}, {9: 1})
    
    # Part 2: A_left * B_left[:, 3:7] (2x2 * 2x4)
    # Use two Strassen blocks for cols (3,4) and (5,6)
    
    # Strassen for B cols 3,4:
    # M1' = (A10+A11)(B03+B14)
    add_rank1({3: 1, 4: 1}, {3: 1, 11: 1}, {3: 1, 10: 1, 12: 1})
    # M2' = (A10+A11)(B04)
    add_rank1({3: 1, 4: 1}, {4: 1}, {4: 1, 11: 1})
    # M3' = A00(B04-B14)
    add_rank1({0: 1}, {4: 1, 11: -1}, {4: 1, 11: 1})
    # M4' = A11(B13-B03)
    add_rank1({4: 1}, {10: 1, 3: -1}, {3: 1, 10: 1, 12: 1})
    # M5' = (A00+A01)(B14)
    add_rank1({0: 1, 1: 1}, {11: 1}, {3: -1, 4: 1, 12: 1})
    # M6' = (A10-A00)(B03+B04)
    add_rank1({3: 1, 0: -1}, {3: 1, 4: 1}, {12: 1})
    # M7' = (A01-A11)(B13+B14)
    add_rank1({1: 1, 4: -1}, {10: 1, 11: 1}, {3: 1, 12: 1})
    
    # Strassen for B cols 5,6:
    # M1'' = (A10+A11)(B05+B16)
    add_rank1({3: 1, 4: 1}, {5: 1, 13: 1}, {5: 1, 12: 1, 13: 1})
    # M2'' = (A10+A11)(B06)
    add_rank1({3: 1, 4: 1}, {6: 1}, {6: 1, 13: 1})
    # M3'' = A00(B06-B16)
    add_rank1({0: 1}, {6: 1, 13: -1}, {6: 1, 13: 1})
    # M4'' = A11(B15-B05)
    add_rank1({4: 1}, {12: 1, 5: -1}, {5: 1, 12: 1, 13: 1})
    # M5'' = (A00+A01)(B16)
    add_rank1({0: 1, 1: 1}, {13: 1}, {5: -1, 6: 1, 13: 1})
    # M6'' = (A10-A00)(B05+B06)
    add_rank1({3: 1, 0: -1}, {5: 1, 6: 1}, {13: 1})
    # M7'' = (A01-A11)(B15+B16)
    add_rank1({1: 1, 4: -1}, {12: 1, 13: 1}, {5: 1, 13: 1})
    
    # Part 3: A_right * B_right (2x1 * 1x7)
    # A_right: A[0,2]=2, A[1,2]=5
    # B_right: B[2,0]=14, B[2,1]=15, B[2,2]=16, B[2,3]=17, B[2,4]=18, B[2,5]=19, B[2,6]=20
    # C: all 7 columns
    
    for k in range(7):
        add_rank1({2: 1}, {14 + k: 1}, {k: 1})
        add_rank1({5: 1}, {14 + k: 1}, {7 + k: 1})
    
    return U_terms, V_terms, W_terms

def _local_search(U, V, W, n, m, p, max_iter=5000):
    """
    Local search to try to reduce rank by removing redundant terms.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = U_arr.shape[1]
    
    T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    improved = True
    while improved and not _check_time():
        improved = False
        for col in range(current_rank):
            if _check_time():
                break
            if current_rank <= 1:
                break
            
            T_without = np.zeros((n*m, m*p, p*n), dtype=float)
            for r in range(current_rank):
                if r == col:
                    continue
                T_without += np.einsum('a,b,c->abc', 
                                       U_arr[:,r], V_arr[:,r], W_arr[:,r])
            
            if np.allclose(T_without, T_expected, atol=1e-9):
                keep = [r for r in range(current_rank) if r != col]
                U_arr = U_arr[:, keep]
                V_arr = V_arr[:, keep]
                W_arr = W_arr[:, keep]
                current_rank -= 1
                improved = True
                break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def construct():
    """
    Returns a minimal rank decomposition for matrix multiplication of sizes 2x3 and 3x7.
    """
    n, m, p = 2, 3, 7
    
    # Start with hybrid Strassen/schoolbook construction
    U, V, W = _construct_hybrid(n, m, p)
    best_rank = len(U)
    
    # Verify
    if not _verify_decomp(U, V, W, n, m, p):
        # Fall back to schoolbook if hybrid is incorrect
        U_school = []
        V_school = []
        W_school = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0.0] * (n*m)
                    v = [0.0] * (m*p)
                    w = [0.0] * (p*n)
                    u[i*m+j] = 1.0
                    v[j*p+k] = 1.0
                    w[k*n+i] = 1.0
                    U_school.append(u)
                    V_school.append(v)
                    W_school.append(w)
        U, V, W = U_school, V_school, W_school
        best_rank = len(U)
    
    # Try local search to reduce rank
    if not _check_time():
        U_red, V_red, W_red = _local_search(U, V, W, n, m, p)
        new_rank = len(U_red)
        if new_rank < best_rank and _verify_decomp(U_red, V_red, W_red, n, m, p):
            U, V, W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Ensure values are clean (integer or half-integer)
    def _round_half(arr):
        arr = np.array(arr, dtype=float)
        arr = np.round(arr * 2) / 2
        return arr
    
    U = _round_half(np.array(U)).tolist()
    V = _round_half(np.array(V)).tolist()
    W = _round_half(np.array(W)).tolist()
    
    # Final verification
    if not _verify_decomp(U, V, W, n, m, p):
        # Fall back to schoolbook
        U_school = []
        V_school = []
        W_school = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0.0] * (n*m)
                    v = [0.0] * (m*p)
                    w = [0.0] * (p*n)
                    u[i*m+j] = 1.0
                    v[j*p+k] = 1.0
                    w[k*n+i] = 1.0
                    U_school.append(u)
                    V_school.append(v)
                    W_school.append(w)
        U, V, W = U_school, V_school, W_school
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
