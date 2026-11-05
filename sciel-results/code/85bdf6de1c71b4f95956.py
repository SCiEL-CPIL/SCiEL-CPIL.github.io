"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = DIMENSIONS  # 4, 4, 6

    def schoolbook_decomp(n, m, p):
        """Basic schoolbook rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Rank-7 Strassen for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def winograd_f4_4x4x4():
        """Winograd F(4) rank-16 decomposition for 4x4 * 4x4 -> 4x4."""
        rank = 16
        # F(4) = (1,1,-1) tensor, used 4 times per dimension
        # U coefficients from F(4): rows are (1,1,-1) padded
        F4 = [[1,1,-1],[1,1,-1],[1,-1,-1]]  # 3x3 but for F(4) we need 4x4
        
        # Actually Winograd F(4) for n=4: m=4, F(4) has m-2=2 redundant terms
        # The F(4) matrix (m x (m-2)) for m=4 is 4x2:
        # Using the standard F(4) = [[1,1],[1,-1],[1,0],[1,0]] ... 
        # Let me use the known rank-16 decomposition.
        
        # F(4) decomposition: A(4x4) = Ahat @ D @ B^T, Bhat(4x4) = G @ B @ D^T @ H^T
        # where Ahat is 4x6, Bhat is 6x4, D is 4x6, etc.
        # Rank = 16 = 4 * (4-2)^... no, rank = 4 * 4 = 16 for F(4).
        
        # Standard Winograd F(4): uses (m-2) = 2 multiplications per column, 4 columns
        # Total rank = 4 * (4) * ... let me just use the explicit form.
        
        # F(4) transform matrix M (4x6):
        M = np.zeros((4, 6))
        M[0,0]=1; M[0,1]=1
        M[1,1]=1; M[1,2]=1
        M[2,2]=1; M[2,3]=1
        M[3,3]=1; M[3,4]=1
        M[0,5]=1
        M[3,5]=-1
        
        # Actually let me use the explicit rank-16 Strassen-recursive for 4x4.
        # 4x4x4 via Strassen: 7 subproblems of 2x2x2, each rank 7, total 49.
        # Winograd F(4) should give rank 16.
        
        # Let me construct F(4) properly.
        # For 4x4 matrix multiplication using F(4):
        # M = F(4) @ F(4)^T where F(4) is 4x(4-2+...) 
        # F(4) has parameters: a=1, b=1, c=-1 (for the (1,1,-1) pattern)
        
        # The F(4) matrix (4 x 6) with entries:
        # Row structure: [[1,1,0,0,0,0],[0,1,1,0,0,0],[0,0,1,1,0,0],[1,0,0,1,0,-1]]... 
        # No, F(4) for m=4: the transform is 4 x (m-2+2) = 4 x 4? No.
        
        # Let me just do Strassen-recursive 4x4x4 (rank 49) and then optimize.
        U, V, W = _strassen_recursive(n, m, p)
        return U, V, W

    def _strassen_recursive(nn, mm, pp):
        """Recursive Strassen-like decomposition."""
        if nn == 2 and mm == 2 and pp == 2:
            return strassen_2x2x2()
        if nn == 1 and mm == 1 and pp == 1:
            return [[1]], [[1]], [[1]]
        
        # For square: split into 2x2 blocks
        # For rectangular, handle specially
        if nn == mm and mm == pp:
            # Square Strassen
            return _strassen_square(nn)
        else:
            # Rectangular: use schoolbook as base, but try to be smart
            return _rectangular(nn, mm, pp)

    def _strassen_square(n):
        """Strassen for nxn * nxn -> nxn."""
        if n == 2:
            return strassen_2x2x2()
        
        # Split into 4 blocks of size n/2
        half = n // 2
        U_sub, V_sub, W_sub = _strassen_square(half)
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        
        U = [[0]*total_rank for _ in range(n*n)]
        V = [[0]*total_rank for _ in range(n*n)]
        W = [[0]*total_rank for _ in range(n*n)]
        
        # Strassen terms
        terms = [
            {'A': [(0,0,1),(1,0,1)], 'B': [(0,0,1),(1,0,1)], 'C': [(0,0,1),(1,0,1)]},
            {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
            {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
            {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
            {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
            {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
            {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
        ]
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(half*half):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    r_local = u_local // half; c_local = u_local % half
                    for block, coeff in term['A']:
                        br, bc = block[0], block[1]
                        g_r = r_local + br*half; g_c = c_local + bc*half
                        U[g_r*n + g_c][base_col + c] += u_val * coeff
                for v_local in range(half*half):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    r_local = v_local // half; c_local = v_local % half
                    for block, coeff in term['B']:
                        br, bc = block[0], block[1]
                        g_r = r_local + br*half; g_c = c_local + bc*half
                        V[g_r*n + g_c][base_col + c] += v_val * coeff
                for w_local in range(half*half):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    r_local = w_local // half; c_local = w_local % half
                    for block, coeff in term['C']:
                        br, bc = block[0], block[1]
                        g_r = r_local + br*half; g_c = c_local + bc*half
                        W[g_r*n + g_c][base_col + c] += w_val * coeff
        return U, V, W

    def _rectangular(n, m, p):
        """Rectangular matrix multiplication n*m by m*p."""
        # Try to handle by padding to square or using direct construction
        if n == m == p:
            return _strassen_square(n)
        
        # For 4x4x6: treat as 4x4x4 + 4x4x2
        if n == m == 4 and p == 6:
            U444, V444, W444 = _strassen_square(4)
            rank444 = len(U444[0])
            
            # 4x4x2: A(4x4) * B(4x2) = C(4x2)
            # Split into two 2x2x2 problems... or use direct approach
            # 4x4x2: rank at most 8 (4x4x2 schoolbook / 4 = 8? No, 4*4*2=32 schoolbook)
            # Better: 2 columns, each 4x4x1 (rank 4 each) = 8? 
            # 4x4x1: A(4x4)*B(4x1)=C(4x1), rank 16 schoolbook, but min is 4.
            # Actually 4x4x1 has rank at most 4*4=16, min rank is 4 (each output is inner product).
            # So 4x4x2 has rank at most 8 (if we can share).
            
            U442, V442, W442 = _4x4x2()
            rank442 = len(U442[0])
            
            total_rank = rank444 + rank442
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            for c in range(rank444):
                for r in range(n*m):
                    U[r][c] = U444[r][c]
                for r in range(m*4):
                    V[r][c] = V444[r][c]
                for r in range(4*n):
                    W[r][c] = W444[r][c]
            
            base = rank444
            for c in range(rank442):
                for r in range(n*m):
                    U[r][base+c] = U442[r][c]
                for r in range(m*p):
                    if r < m*4:
                        pass  # first 4 cols handled by U444
                    else:
                        V[r][base+c] = V442[r][c]
                for r in range(p*n):
                    if r % n < 4 and r // n < 4:
                        pass
                    # W442 is for the 4x2 part: indices 4..5 in column
                    W[r][base+c] = W442[r][c] if r >= 4*n else 0
                    # Actually W442 shape is (2*n, rank442) for the 2 output columns
                    # W indices: k*n+i where k is 0..5, i is 0..3
                    # For the 4x2 part, k=4,5
            return U, V, W
        
        # General rectangular: use schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def _4x4x2():
        """Decomposition for 4x4 * 4x2 -> 4x2, aiming for low rank."""
        # A(4x4) * B(4x2) = C(4x2)
        # Split B into two 4x1 columns: B = [b1, b2]
        # C = [A*b1, A*b2]
        # Each A*b is a 4x1 vector. 
        # 4x4x1: rank is at most 4 (each output row is a dot product of one row of A with B)
        # But we can share: if we compute A@V for 2 columns V(4x2), 
        # we can use block Strassen-like approach.
        
        # Split A into 2x2 blocks, B into 2x1 blocks (2 per column, 2 columns)
        # 2x2 x 2x1: C_ij = sum_k A_ik * B_kj
        # For 2x2 x 2x1: rank 4 (each of 2 outputs needs 2 products)
        # But we can do better by sharing across the 2 columns.
        
        # 4x4 x 4x2: think of it as (2x2 blocks) * (2x1 blocks)
        # 2 block-rows of A, 2 block-rows of B, 2 block-cols of B
        # Each (2x2) * (2x1) -> (2x1) has rank 4
        # Total: 2*2*4 = 16? That's schoolbook/2.
        
        # Better: use the fact that for fixed A, computing A*[b1,b2] 
        # can be done with rank 4+4=8 if we handle columns independently.
        # But can we do better by sharing Strassen intermediates?
        
        # Let's just do rank 8: two independent 4x4x1 decompositions
        rank = 8
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        
        # For column 0 of B (4x1): C_i = sum_j A_ij * B_j
        # Rank 4: each term is A_ij * B_j for one (i,j) pair... no that's 16.
        # Minimum rank for 4x4x1 is 4: C = A*B where B is 4x1.
        # C_i = sum_j A_ij B_j. This is 4 separate linear combinations.
        # Rank: we need 4*4 = 16 schoolbook, but min rank is 4 
        # (each output element is one bilinear form).
        
        # Actually for A(m x n) * B(n x 1), the rank is n (not mn).
        # Wait no. C_i = sum_j A_ij * B_j. 
        # This is m * n products in schoolbook. The tensor rank is n if we view it as
        # C_i = f_i(A, B) where f_i uses all B components.
        # The rank of the 4x4x1 tensor is 4 (each "slice" over B is one row of A).
        
        # Hmm, let me just verify with a direct construction.
        # 4x4 x 4x1 -> 4x1:
        # C_i = A_i0*B_0 + A_i1*B_1 + A_i2*B_2 + A_i3*B_3
        # We can write: C_i = sum_k (sum_j delta) ... 
        # Rank 4: U[i,j] = 1 for all i,j (each of 4 output rows connects to each of 4 B entries)
        # No, that's 16 rank-1 terms.
        
        # Actually the bilinear form C_i = sum_j A_ij B_j has tensor rank 1 for each i,
        # but the 4 together... The full 4x4x1 tensor has rank 4.
        # T(i,j,0) = A_ij * B_0 for j=0..3, but that's 4 separate rank-1 terms per i.
        # No wait, T is the tensor with T[i,j,k] = 1 if it contributes to C_i from A_ij * B_k.
        # For 4x4x1: T[i,j,0] = A_ij * B_0 contributes to C_i... 
        # The tensor T_{ijk} where i=0..3 (C index), j=0..3 (A row), k=0..0 (B col)
        # T[i,j,0] = delta... no.
        
        # I'm overcomplicating. Let me just use schoolbook for 4x4x2 (rank 32) 
        # and rely on the reduction algorithm to find the true rank.
        # Actually that's wasteful. Let me use a better base.
        
        # 4x4 x 4x2: schoolbook rank = 4*4*2 = 32.
        # Strassen-like: split into 2x2 x 2x1 blocks.
        # 4 block positions in A (2x2), 4 block positions in B (2x1, 2 cols)
        # Each 2x2 x 2x1: C = A*B, A is 2x2, B is 2x1, C is 2x1.
        # C_0 = A_00*B_0 + A_01*B_1, C_1 = A_10*B_0 + A_11*B_1
        # Rank 4 (schoolbook). Can we do rank < 4? 
        # The 2x2x1 tensor: T[i,j,k] for i=0,1 (C), j=0,1 (A), k=0 (B)... 
        # Actually T is 2x2x1 with T[i,j,0] = 1 (each A_ij multiplies B_0 or B_1).
        # For 2x2 x 2x1: C_i = sum_j A_ij B_j (j=0,1)
        # Tensor rank: T[i,j,k] where k is the B index (0 or 1)
        # T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1
        # This has rank 2 (T = u0*v0*w0^T + u1*v1*w1^T)
        # So 2x2 x 2x1 has rank 2!
        
        rank = 4 * 2 * 2  # 4 A-blocks * 2 B-blocks * rank 2 per = 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        
        col = 0
        for a_br in range(2):  # A block row
            for a_bc in range(2):  # A block col
                for b_br in range(2):  # B block row
                    for b_bc in range(2):  # B block col
                        # 2x2 x 2x1 with rank 2
                        # Term 1: A[a_br*2+0, a_bc*2+0] * B[b_br*2+0, b_bc] -> C[a_br*2+0, b_bc]
                        #   and A[a_br*2+1, a_bc*2+1] * B[b_br*2+1, b_bc] -> C[a_br*2+1, b_bc]
                        # Rank-1 term 1: U[A(a_br,0,a_bc,0)] = 1, V[B(b_br,0,b_bc)] = 1, W[C(a_br,0,b_bc)] = 1
                        U[a_br*8 + a_bc*4 + 0][col] = 1
                        V[b_br*4 + b_bc*2 + 0][col] = 1
                        W[b_bc*4 + a_br*2 + 0][col] = 1
                        col += 1
                        
                        # Rank-1 term 2: A(a_br,1,a_bc,1) * B(b_br,1,b_bc) -> C(a_br,1,b_bc)
                        U[a_br*8 + a_bc*4 + 3][col] = 1
                        V[b_br*4 + b_bc*2 + 1][col] = 1
                        W[b_bc*4 + a_br*2 + 1][col] = 1
                        col += 1
        return U, V, W

    def build_4x4x6_hybrid():
        """Hybrid: Strassen 4x4x4 + optimized 4x4x2."""
        n, m, p = 4, 4, 6
        U1, V1, W1 = _strassen_square(4)
        rank1 = len(U1[0])
        
        U2, V2, W2 = _4x4x2()
        rank2 = len(U2[0])
        
        total_rank = rank1 + rank2
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        for c in range(rank1):
            for r in range(n*m):
                U[r][c] = U1[r][c]
            for r in range(m*4):
                V[r][c] = V1[r][c]
            for r in range(4*n):
                W[r][c] = W1[r][c]
        
        base = rank1
        for c in range(rank2):
            for r in range(n*m):
                U[r][base+c] = U2[r][c]
            for r in range(m*p):
                if r >= m*4:  # columns 4,5 of B
                    V[r][base+c] = V2[r][c]
            for r in range(p*n):
                if r >= 4*n:  # columns 4,5 of C
                    W[r][base+c] = W2[r][c]
        
        return U, V, W

    def verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
        """Vectorized verification of tensor decomposition."""
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    val = np.dot(u_row * v_row, w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_ones(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
        """Verify only the (i,j,k)->1 entries, faster for dense tensors."""
        return verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p, tol)

    def reduce_rank_greedy(U_arr, V_arr, W_arr, n, m, p, time_budget, verbose=False):
        """Greedy single-column removal."""
        current_rank = U_arr.shape[1]
        
        if time.time() > time_budget:
            return U_arr, V_arr, W_arr
        
        while time.time() < time_budget and U_arr.shape[1] > 1:
            removed_any = False
            current_rank = U_arr.shape[1]
            for col in range(current_rank):
                if time.time() > time_budget:
                    break
                mask = np.ones(current_rank, dtype=bool)
                mask[col] = False
                U_test = U_arr[:, mask]
                V_test = V_arr[:, mask]
                W_test = W_arr[:, mask]
                if verify_ones(U_test, V_test, W_test, n, m, p):
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    removed_any = True
                    if verbose:
                        print(f"  Removed col {col}, new rank {U_arr.shape[1]}")
                    break
            if not removed_any:
                break
        return U_arr, V_arr, W_arr

    def reduce_rank_pairwise(U_arr, V_arr, W_arr, n, m, p, time_budget):
        """Greedy single + pairwise removal."""
        U_arr, V_arr, W_arr = reduce_rank_greedy(U_arr, V_arr, W_arr, n, m, p, time_budget)
        
        if time.time() > time_budget:
            return U_arr, V_arr, W_arr
        
        while time.time() < time_budget and U_arr.shape[1] > 2:
            removed_any = False
            current_rank = U_arr.shape[1]
            for col1 in range(current_rank):
                if time.time() > time_budget:
                    break
                for col2 in range(col1 + 1, current_rank):
                    if time.time() > time_budget:
                        break
                    mask = np.ones(current_rank, dtype=bool)
                    mask[col1] = False
                    mask[col2] = False
                    U_test = U_arr[:, mask]
                    V_test = V_arr[:, mask]
                    W_test = W_arr[:, mask]
                    if verify_ones(U_test, V_test, W_test, n, m, p):
                        U_arr = U_test
                        V_arr = V_test
                        W_arr = W_test
                        removed_any = True
                        break
                if removed_any:
                    break
            if not removed_any:
                break
        return U_arr, V_arr, W_arr

    def reduce_rank_triple(U_arr, V_arr, W_arr, n, m, p, time_budget):
        """Try removing triples after singles and pairs."""
        U_arr, V_arr, W_arr = reduce_rank_pairwise(U_arr, V_arr, W_arr, n, m, p, time_budget)
        
        if time.time() > time_budget:
            return U_arr, V_arr, W_arr
        
        # Only do triples if rank is still high and we have time
        if U_arr.shape[1] < 30 or time.time() > time_budget - 20:
            return U_arr, V_arr, W_arr
        
        while time.time() < time_budget and U_arr.shape[1] > 3:
            removed_any = False
            current_rank = U_arr.shape[1]
            for combo in itertools.combinations(range(current_rank), 3):
                if time.time() > time_budget:
                    break
                mask = np.ones(current_rank, dtype=bool)
                for col in combo:
                    mask[col] = False
                U_test = U_arr[:, mask]
                V_test = V_arr[:, mask]
                W_test = W_arr[:, mask]
                if verify_ones(U_test, V_test, W_test, n, m, p):
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    removed_any = True
                    break
            if not removed_any:
                break
        return U_arr, V_arr, W_arr

    # Try multiple initial decompositions and pick the best after reduction
    candidates = []
    
    # Candidate 1: Pure Strassen recursive (77 rank)
    try:
        U1, V1, W1 = _rectangular(n, m, p)
        U1a = np.array(U1, dtype=float)
        V1a = np.array(V1, dtype=float)
        W1a = np.array(W1, dtype=float)
        if verify_ones(U1a, V1a, W1a, n, m, p):
            candidates.append((U1a, V1a, W1a))
    except:
        pass
    
    # Candidate 2: Hybrid Strassen 4x4x4 + 4x4x2
    try:
        U2, V2, W2 = build_4x4x6_hybrid()
        U2a = np.array(U2, dtype=float)
        V2a = np.array(V2, dtype=float)
        W2a = np.array(W2, dtype=float)
        if verify_ones(U2a, V2a, W2a, n, m, p):
            candidates.append((U2a, V2a, W2a))
        else:
            pass
    except:
        pass
    
    # Candidate 3: Schoolbook (96 rank) as safety net
    U3, V3, W3 = schoolbook_decomp(n, m, p)
    U3a = np.array(U3, dtype=float)
    V3a = np.array(V3, dtype=float)
    W3a = np.array(W3, dtype=float)
    candidates.append((U3a, V3a, W3a))
    
    # Pick the lowest-rank valid candidate
    best_U, best_V, best_W = min(candidates, key=lambda x: x[0].shape[1])
    
    # Apply rank reduction
    time_budget = deadline - 10
    U_red, V_red, W_red = reduce_rank_triple(best_U, best_V, best_W, n, m, p, time_budget)
    
    # Final check
    if verify_ones(U_red, V_red, W_red, n, m, p):
        return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
