"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Uses a recursive Strassen-based approach to minimize the rank.
    The base case for 2x2x2 uses the known optimal rank 7 (Strassen).
    For other dimensions, we split and recurse.
    
    The output follows the trace convention: C[k*n + i, :] = sum_j A[i*m+j, :] * B[j*p+k, :]
    which corresponds to the tensor T[i,j,k] = A[i,j] * B[j,k].
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    # We will build the decomposition recursively.
    # The state is (n, m, p) and we return lists of (U_col, V_col, W_col)
    # where U_col is a vector of length n*m, V_col is a vector of length m*p,
    # and W_col is a vector of length p*n.
    
    # To avoid recomputation, we use memoization.
    # However, the indices of the sub-problems depend on the split, so we need to be careful.
    # For simplicity and given the small size, we might just compute directly.
    # But let's try to define a recursive function that generates the columns.
    
    # Since the dimensions are small, we can use a direct construction for small cases
    # and recursion for larger ones.
    
    # Let's define a function that, given a sub-problem defined by ranges of indices,
    # generates the rank-1 terms.
    
    # Actually, a cleaner way is to use the algebraic properties of tensor products.
    # If we have a decomposition for (n1, m1, p1) and (n2, m2, p2),
    # the tensor product has rank R1 * R2.
    
    # For 2x4x7, we can decompose it as (2x2x7) x (1x2x1) or similar.
    # Or (2x4x2) x (1x1x3) + (2x4x1) ? No, that's not right.
    
    # Let's use a recursive split on the middle dimension m.
    # Split m into m1 and m2.
    # The tensor T(i, j, k) = A(i, j) B(j, k).
    # If we split j into j1 in [0, m1) and j2 in [m1, m),
    # T(i, j, k) = sum_{j1} A(i, j1) B(j1, k) + sum_{j2} A(i, j2) B(j2, k).
    # This is a sum of two tensors: T1(i, j1, k) and T2(i, j2, k).
    # T1 is effectively a (n, m1, p) tensor but with j1 indexed locally.
    # T2 is a (n, m-m1, p) tensor.
    # The rank of the sum is at most rank(T1) + rank(T2).
    
    # However, this direct sum doesn't exploit Strassen's structure.
    # Strassen exploits the structure when n=m=p=2.
    
    # Let's try to split 2x4x7 into smaller blocks where Strassen can be applied.
    # 2x4x7 = (2x2x7) + (2x2x7) ? No, m=4, so we can split m into 2+2.
    # T(i, j, k) = sum_{j=0}^3 A(i,j) B(j,k).
    # Let j=0,1 be part 1, j=2,3 be part 2.
    # Part 1: A1 is 2x2 (cols 0,1), B1 is 2x7 (rows 0,1). T1(i, j1, k) = A1(i, j1) B1(j1, k).
    # This is a 2x2x7 tensor.
    # Part 2: A2 is 2x2 (cols 2,3), B2 is 2x7 (rows 2,3). T2(i, j2, k) = A2(i, j2) B2(j2, k).
    # This is also a 2x2x7 tensor.
    
    # Now, how to decompose 2x2x7 with minimal rank?
    # 2x2x7 = 2 * (1x2x7) ? No.
    # We can view 2x2x7 as the tensor product of (2x2x2) and (1x1x3) plus remainder?
    # Or simply, since 7 = 2+2+2+1, we can split k into 2, 2, 2, 1.
    # Then 2x2x7 = (2x2x2) + (2x2x2) + (2x2x2) + (2x2x1).
    # Rank of 2x2x2 is 7 (Strassen).
    # Rank of 2x2x1 is 4 (schoolbook, since p=1, rank is n*m*p = 4).
    # So total rank for 2x2x7 would be 3*7 + 4 = 25.
    
    # Then total rank for 2x4x7 would be 2 * 25 = 50.
    
    # Can we do better?
    # Maybe 2x2x7 can be done with rank < 25?
    # The rank of 2x2x7 is known to be at most 7*ceil(7/2) ? No.
    # Actually, the rank of 2x2xn is known for small n.
    # For 2x2x3, rank is 7? Or 8?
    # Let's stick with the safe recursive construction first.
    
    # Recursive construction plan:
    # 1. Base cases:
    #    - 1x1x1: rank 1
    #    - 1xMxP: rank M*P (diagonal)
    #    - Nx1xP: rank N*P
    #    - NxMx1: rank N*M
    #    - 2x2x2: rank 7 (Strassen)
    #    - 2x2x1: rank 4
    #    - 2x2x3: rank 7? (Check if we can reuse Strassen-like)
    #      Actually, 2x2x3 = 2x2x2 + 2x2x1, rank <= 7+4=11.
    #      But is there a better one?
    # 2. Recursive step:
    #    - Split the largest dimension.
    #    - If n is even, split n into n/2, n/2.
    #    - If m is even, split m into m/2, m/2.
    #    - If p is even, split p into p/2, p/2.
    #    - Combine the ranks.
    
    # For 2x4x7:
    # Split m=4 into 2+2.
    # Sub-problems: 2x2x7 and 2x2x7.
    # For 2x2x7:
    # Split p=7 into 2+2+2+1.
    # Sub-problems: 2x2x2, 2x2x2, 2x2x2, 2x2x1.
    # Ranks: 7, 7, 7, 4. Sum = 25.
    # Total rank = 25 + 25 = 50.
    
    # Let's implement this.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # We need to handle the index mapping correctly.
    # The global indices are:
    # A: i in [0, n), j in [0, m)
    # B: j in [0, m), k in [0, p)
    # C: k in [0, p), i in [0, n)
    
    # When we split, we need to map local indices to global indices.
    
    def solve(n, m, p, i_offset, j_offset, k_offset):
        """
        Solves the sub-problem for A[i_offset:i_offset+n, j_offset:j_offset+m]
        multiplied by B[j_offset:j_offset+m, k_offset:k_offset+p].
        Appends rank-1 terms to U_cols, V_cols, W_cols.
        """
        # Base cases
        if n == 1 and m == 1 and p == 1:
            u = [0] * (2*4) # n*m for the whole problem? No, we need the global size.
            # Wait, the U, V, W matrices must have dimensions based on the ORIGINAL n, m, p.
            # U: (n_total * m_total, rank)
            # V: (m_total * p_total, rank)
            # W: (p_total * n_total, rank)
            # So we need to know the total n, m, p.
            # Let's pass them or use global variables.
            pass
        
    # Let's use global variables for total dimensions.
    total_n, total_m, total_p = n, m, p
    
    def solve_sub(n, m, p, i_off, j_off, k_off):
        if n == 0 or m == 0 or p == 0:
            return
        
        # Check deadline
        if time.time() > deadline - 10:
            return
        
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            i = i_off
            j = j_off
            k = k_off
            u = [0] * (total_n * total_m)
            v = [0] * (total_m * total_p)
            w = [0] * (total_p * total_n)
            u[i * total_m + j] = 1
            v[j * total_p + k] = 1
            w[k * total_n + i] = 1
            add_term(u, v, w)
            return
        
        # Base case: 2x2x2 (Strassen)
        if n == 2 and m == 2 and p == 2:
            # Indices
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # Strassen terms
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We need to generate the U, V, W vectors for each Mx.
            # Ux contains the coefficients for A.
            # Vx contains the coefficients for B.
            # Wx contains the coefficients for C.
            
            def make_uv_w(a_coeffs, b_coeffs, c_coeffs):
                """
                a_coeffs: list of (i, j, val)
                b_coeffs: list of (j, k, val)
                c_coeffs: list of (k, i, val)
                """
                u = [0] * (total_n * total_m)
                v = [0] * (total_m * total_p)
                w = [0] * (total_p * total_n)
                
                for (i, j, val) in a_coeffs:
                    u[i * total_m + j] = val
                for (j, k, val) in b_coeffs:
                    v[j * total_p + k] = val
                for (k, i, val) in c_coeffs:
                    w[k * total_n + i] = val
                return u, v, w
            
            # M1
            a1 = [((i0, j0, 1), (i1, j1, 1))]
            b1 = [((j0, k0, 1), (j1, k1, 1))]
            c1 = [((k0, i0, 1), (k1, i1, -1))] # Wait, C00 gets +M1, C11 gets +M1?
            # C00 = M1 + ... => C(k0, i0) += M1
            # C11 = M1 - ... => C(k1, i1) += M1
            # So W for M1 has +1 at (k0, i0) and +1 at (k1, i1).
            c1 = [((k0, i0, 1), (k1, i1, 1))]
            add_term(*make_uv_w(a1, b1, c1))
            
            # M2
            a2 = [((i1, j0, 1), (i1, j1, 1))]
            b2 = [((j0, k0, 1))]
            c2 = [((k0, i1, 1))] # C10 = M2 + ...
            add_term(*make_uv_w(a2, b2, c2))
            
            # M3
            a3 = [((i0, j0, 1))]
            b3 = [((j0, k1, 1), (j1, k1, -1))] # B01 - B11
            c3 = [((k1, i0, 1))] # C01 = M3 + ...
            add_term(*make_uv_w(a3, b3, c3))
            
            # M4
            a4 = [((i1, j1, 1))]
            b4 = [((j1, k0, 1), (j0, k0, -1))] # B10 - B00
            c4 = [((k0, i0, 1)), ((k0, i1, 1))] # C00 += M4, C10 += M4
            add_term(*make_uv_w(a4, b4, c4))
            
            # M5
            a5 = [((i0, j0, 1), (i0, j1, 1))]
            b5 = [((j1, k1, 1))]
            c5 = [((k1, i0, -1)), ((k1, i1, 1))] # C00 -= M5, C11 += M5
            add_term(*make_uv_w(a5, b5, c5))
            
            # M6
            a6 = [((i1, j0, 1), (i0, j0, -1))]
            b6 = [((j0, k0, 1), (j0, k1, 1))]
            c6 = [((k1, i1, 1))] # C11 += M6
            add_term(*make_uv_w(a6, b6, c6))
            
            # M7
            a7 = [((i0, j1, 1), (i1, j1, -1))]
            b7 = [((j1, k0, 1), (j1, k1, 1))]
            c7 = [((k0, i0, 1))] # C00 += M7
            add_term(*make_uv_w(a7, b7, c7))
            
            return
        
        # Recursive steps
        # Try to split the largest dimension
        # Split n if n > 1 and even
        if n > 1 and n % 2 == 0:
            n1 = n // 2
            solve_sub(n1, m, p, i_off, j_off, k_off)
            solve_sub(n1, m, p, i_off + n1, j_off, k_off)
            return
        
        # Split m if m > 1 and even
        if m > 1 and m % 2 == 0:
            m1 = m // 2
            solve_sub(n, m1, p, i_off, j_off, k_off)
            solve_sub(n, m1, p, i_off, j_off + m1, k_off)
            return
        
        # Split p if p > 1 and even
        if p > 1 and p % 2 == 0:
            p1 = p // 2
            solve_sub(n, m, p1, i_off, j_off, k_off)
            solve_sub(n, m, p1, i_off, j_off, k_off + p1)
            return
        
        # If odd, split into 1 and rest
        # Split n
        if n > 1:
            solve_sub(1, m, p, i_off, j_off, k_off)
            solve_sub(n-1, m, p, i_off + 1, j_off, k_off)
            return
        
        # Split m
        if m > 1:
            solve_sub(n, 1, p, i_off, j_off, k_off)
            solve_sub(n, m-1, p, i_off, j_off + 1, k_off)
            return
        
        # Split p
        if p > 1:
            solve_sub(n, m, 1, i_off, j_off, k_off)
            solve_sub(n, m, p-1, i_off, j_off, k_off + 1)
            return
    
    # Start the recursion
    solve_sub(n, m, p, 0, 0, 0)
    
    # Construct the final matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(total_n * total_m)]
    V = [[0] * rank for _ in range(total_m * total_p)]
    W = [[0] * rank for _ in range(total_p * total_n)]
    
    for c in range(rank):
        for i in range(total_n * total_m):
            U[i][c] = U_cols[c][i]
        for j in range(total_m * total_p):
            V[j][c] = V_cols[c][j]
        for k in range(total_p * total_n):
            W[k][c] = W_cols[c][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
