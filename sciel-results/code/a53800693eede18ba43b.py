"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline with safety margin
    
    # Schoolbook as fallback - rank = n*m*p = 72
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def check_decomposition(U, V, W, rank, n, m, p):
        """Check if U, V, W of given rank satisfy the decomposition."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    def strassen_decompose_2x2x2():
        """
        Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        # Strassen's 7 products for 2x2 matrices
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For tensor decomposition, we need U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
        # such that sum_r equals 1 for the corresponding (i,j,k) entry.
        
        # The 7 bilinear forms:
        # r=0: M1 = (a+d)(e+h) -> U[a]=1,U[d]=1; V[e]=1,V[h]=1; W contributions to C11,C22
        # Let's think of this differently.
        
        # Actually for matrix multiplication tensor T[i,j,k] = delta where C[i,k] = sum_j A[i,j]*B[j,k]
        # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
        
        # Strassen's algorithm computes 7 products:
        # p1 = (A11+A22)(B11+B22)
        # p2 = (A21+A22)B11
        # p3 = A11(B21-B11)
        # p4 = A22(B12-B22)
        # p5 = (A11+A12)B22
        # p6 = (A21-A11)(B11+B12)
        # p7 = (A12-A22)(B21+B22)
        # C11 = p1+p4-p5+p7
        # C12 = p3+p5
        # C21 = p2+p4
        # C22 = p1-p2+p3+p6
        
        # So the decomposition has rank 7 with:
        # For each product pr, we have coefficients in A, B, and C.
        
        U = [[0]*7 for _ in range(4)]  # U[i*2+j, r]
        V = [[0]*7 for _ in range(4)]  # V[j*2+k, r]
        W = [[0]*7 for _ in range(4)]  # W[k*2+i, r]
        
        # r=0: p1 = (A11+A22)(B11+B22), contributes to C11(+1), C22(+1)
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[1][0] = 1  # B12... wait, B11 is B[0,0], B22 is B[1,1]
        # Let me re-index: A[i,j] with i,j in {0,1}, B[j,k] with j,k in {0,1}, C[i,k]
        # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
        # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
        # C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1]
        
        # U[i*2+j, r]: i is row of A, j is col of A
        # V[j*2+k, r]: j is row of B, k is col of B
        # W[k*2+i, r]: k is col of C (which is col of B), i is row of C (which is row of A)
        
        # r=0: p1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # U[0*2+0][0] = 1, U[1*2+1][0] = 1
        # V[0*2+0][0] = 1, V[1*2+1][0] = 1
        # C[0,0] += p1, C[1,1] += p1
        # W[0*2+0][0] = 1, W[1*2+1][0] = 1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: p2 = (A[1,0]+A[1,1])B[0,0]
        # U[1*2+0][1] = 1, U[1*2+1][1] = 1
        # V[0*2+0][1] = 1
        # C[1,0] += p2
        # W[0*2+1][1] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # W[k*2+i] where k=0, i=1 -> W[0*2+1] = W[1]... 
        # Wait, W index is k*2+i. C[i,k] so i is row, k is col.
        # C[1,0] means i=1, k=0, so W[0*2+1] = W[1]
        # Let me redo: W[k*2+i, r]
        # For C[i,k], the W index is k*2+i
        # C[1,0]: i=1, k=0 -> W[0*2+1] = W[1]
        W[1][1] = 1
        
        # r=2: p3 = A[0,0](B[1,0]-B[0,0])
        # U[0*2+0][2] = 1
        # V[1*2+0][2] = 1, V[0*2+0][2] = -1
        # C[0,1] += p3
        # C[0,1]: i=0, k=1 -> W[1*2+0] = W[2]
        U[0][2] = 1
        V[2][2] = 1; V[0][2] = -1
        W[2][2] = 1
        
        # r=3: p4 = A[1,1](B[0,1]-B[1,1])
        # U[1*2+1][3] = 1
        # V[0*2+1][3] = 1, V[1*2+1][3] = -1
        # C[0,0] += p4, C[1,0] += p4
        # C[0,0]: i=0,k=0 -> W[0*2+0] = W[0]
        # C[1,0]: i=1,k=0 -> W[0*2+1] = W[1]
        U[3][3] = 1
        V[1][3] = 1; V[3][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: p5 = (A[0,0]+A[0,1])B[1,1]
        # U[0*2+0][4] = 1, U[0*2+1][4] = 1
        # V[1*2+1][4] = 1
        # C[0,1] -= p5, C[0,0] -= p5
        # C[0,1]: i=0,k=1 -> W[1*2+0] = W[2]
        # C[0,0]: i=0,k=0 -> W[0*2+0] = W[0]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = -1; W[0][4] = -1
        
        # r=5: p6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        # U[1*2+0][5] = 1, U[0*2+0][5] = -1
        # V[0*2+0][5] = 1, V[0*2+1][5] = 1
        # C[1,1] += p6
        # C[1,1]: i=1,k=1 -> W[1*2+1] = W[3]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: p7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        # U[0*2+1][6] = 1, U[1*2+1][6] = -1
        # V[1*2+0][6] = 1, V[1*2+1][6] = 1
        # C[0,0] += p7
        # C[0,0]: i=0,k=0 -> W[0*2+0] = W[0]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def build_3x3x8_from_strassen():
        """
        Build a decomposition for 3x3 x 3x8 using Strassen-like ideas.
        
        Strategy: Partition the 3x3 x 3x8 into blocks and use rank-7 for 2x2x2 parts.
        
        Let A be 3x3, B be 3x8.
        C = A * B, which is 3x8.
        
        We can partition A into 2x2 blocks with overlaps, or use a different approach.
        
        Alternative: Use the fact that 3x3x3 has rank at most 23 (or lower with better algorithms).
        For 3x3x8, we can think of it as 3x3x3 + 3x3x5.
        
        Actually, let's try a more systematic approach:
        Use Strassen's algorithm to compute 2x2 subproducts, and handle the remaining entries.
        
        Let's partition:
        A = [[A11, A12, A13],
             [A21, A22, A23],
             [A31, A32, A33]]
        B = [[B11, B12, ..., B18],
             [B21, B22, ..., B28],
             [B31, B32, ..., B38]]
        
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For the 2x2 submatrix A[0:2, 0:2] and B[0:2, 0:8], we can use Strassen.
        But Strassen is for 2x2 x 2x2. For 2x2 x 2x8, we can apply Strassen column by column
        or in groups.
        
        Let me try a different approach: direct construction using known results.
        
        The tensor rank of 3x3x3 is known to be 23 (from Landsberg's work).
        For 3x3x8, the rank is at most 3*8 + 2*3*5 = 24 + 30 = 54 if we use a naive combination.
        
        Actually, a better bound: use rank-7 for 2x2x2 and handle the rest.
        
        Let me try to construct explicitly:
        
        Partition the 3x3 matrix A into:
        - A[0:2, 0:2] (2x2 block)
        - A[0:2, 2] (2x1 column)
        - A[2, 0:2] (1x2 row)
        - A[2, 2] (scalar)
        
        Similarly for B (3x8):
        - B[0:2, :] (2x8)
        - B[2, :] (1x8)
        
        C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2] * B[2, :]
        C[2, :] = A[2, 0:2] * B[0:2, :] + A[2, 2] * B[2, :]
        
        For A[0:2, 0:2] * B[0:2, :], this is a 2x2 times 2x8 multiplication.
        We can apply Strassen's algorithm for each pair of columns of B.
        B[0:2, :] has 8 columns, so we can group them into 4 pairs of 2 columns.
        Each 2x2 x 2x2 subproblem has rank 7.
        Total for this part: 4 * 7 = 28.
        
        For A[0:2, 2] * B[2, :], this is a 2x1 times 1x8 multiplication.
        C[0, k] += A[0, 2] * B[2, k]
        C[1, k] += A[1, 2] * B[2, k]
        This requires 2 * 8 = 16 rank-1 terms.
        
        For A[2, 0:2] * B[0:2, :], this is a 1x2 times 2x8 multiplication.
        C[2, k] += A[2, 0] * B[0, k] + A[2, 1] * B[1, k]
        This requires 2 * 8 = 16 rank-1 terms.
        
        For A[2, 2] * B[2, :], this is a 1x1 times 1x8 multiplication.
        C[2, k] += A[2, 2] * B[2, k]
        This requires 8 rank-1 terms.
        
        Total: 28 + 16 + 16 + 8 = 68.
        
        That's worse than schoolbook (72) only slightly. Let me try to do better.
        
        Actually, for the 2x2 x 2x8 part, we can be smarter.
        Instead of treating each pair of columns separately, we can use a single Strassen-like
        decomposition for the entire 2x2 x 2x8 block.
        
        For 2x2 x 2x8:
        A = [[a, b], [c, d]]
        B = [[e1, e2, ..., e8], [f1, f2, ..., f8]]
        C = [[a*e_k + b*f_k for k in 1..8], [c*e_k + d*f_k for k in 1..8]]
        
        We can use Strassen's 7 products applied to the full B:
        p1 = (a+d) * (e_k + f_k) for all k -> 8 products, but they share the same a+d factor
        Actually, p1 = (a+d) * sum over k... no, each C entry is separate.
        
        Let me think again. For 2x2 x 2x8, the tensor has:
        - 4 * 8 = 32 entries in C
        - 4 entries in A
        - 16 entries in B
        
        The rank is at most 4 * 8 = 32 (schoolbook) but can be lower.
        
        Using Strassen's idea:
        p1_k = (a+d)(e_k+f_k) for k=1..8
        p2_k = (c+d)e_k for k=1..8
        p3_k = a(f_k-e_k) for k=1..8
        p4_k = d(e_k-f_k) for k=1..8
        p5_k = (a+b)f_k for k=1..8
        p6_k = (c-a)(e_k+f_k) for k=1..8
        p7_k = (b-d)(f_k+e_k) for k=1..8
        
        But this gives 7*8 = 56 products, which is worse than schoolbook (32).
        
        The issue is that Strassen's algorithm is designed for 2x2 x 2x2 where the output
        is also 2x2. For 2x2 x 2x8, the output is 2x8, so we need more products.
        
        Let me try a different approach: use the known rank-23 decomposition for 3x3x3
        and combine with schoolbook for the remaining 5 columns.
        
        Rank for 3x3x3: 23 (if we can find it)
        Rank for 3x3x5 (schoolbook): 45
        Total: 68
        
        Or use rank-7 for 2x2x2 parts more cleverly.
        
        Actually, let me just try to find a good decomposition through search,
        starting from a reasonable initialization.
        """
        pass
    
    # Let's try to construct a better decomposition using a hybrid approach
    # Use Strassen for 2x2x2 subblocks and schoolbook for the rest
    
    def construct_hybrid_decomposition():
        """
        Construct a decomposition using Strassen for 2x2x2 blocks.
        
        Partition:
        A (3x3) into:
        - Top-left 2x2: A[0:2, 0:2]
        - Top-right 2x1: A[0:2, 2]
        - Bottom-left 1x2: A[2, 0:2]
        - Bottom-right 1x1: A[2, 2]
        
        B (3x8) into:
        - Top 2x8: B[0:2, :]
        - Bottom 1x8: B[2, :]
        
        C (3x8):
        C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2] * B[2, :]
        C[2, :] = A[2, 0:2] * B[0:2, :] + A[2, 2] * B[2, :]
        
        For A[0:2, 0:2] * B[0:2, :], use Strassen for each pair of columns.
        B[0:2, :] has 8 columns, group into 4 pairs: (0,1), (2,3), (4,5), (6,7)
        Each 2x2 x 2x2 subproblem uses rank 7.
        Total: 4 * 7 = 28
        
        For A[0:2, 2] * B[2, :]: 2 * 8 = 16 rank-1 terms
        For A[2, 0:2] * B[0:2, :]: 2 * 8 = 16 rank-1 terms
        For A[2, 2] * B[2, :]: 8 rank-1 terms
        
        Total: 28 + 16 + 16 + 8 = 68
        """
        U = [[0] * 68 for _ in range(9)]
        V = [[0] * 68 for _ in range(24)]
        W = [[0] * 68 for _ in range(24)]
        
        col = 0
        # Strassen for A[0:2,0:2] * B[0:2, :] with column pairs
        for pair_idx in range(4):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1
            U_str, V_str, W_str = strassen_decompose_2x2x2()
            for r in range(7):
                # U indices: A[0:2, 0:2] -> U[0*3+0], U[0*3+1], U[1*3+0], U[1*3+1]
                # U_str indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
                # Map to U_full: A[i,j] -> U[i*3+j]
                for ui in range(4):
                    U[ui * 3 + (ui % 2)][col + r] = U_str[ui][r]  # This mapping is wrong
                # Let me fix the mapping
                # U_str[0] = A[0,0] -> U_full[0*3+0] = U_full[0]
                # U_str[1] = A[0,1] -> U_full[0*3+1] = U_full[1]
                # U_str[2] = A[1,0] -> U_full[1*3+0] = U_full[3]
                # U_str[3] = A[1,1] -> U_full[1*3+1] = U_full[4]
                
                # V_str indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
                # Map to V_full: B[j,k] -> V_full[j*8+k]
                # V_str[0] = B[0,k1] -> V_full[0*8+k1]
                # V_str[1] = B[0,k2] -> V_full[0*8+k2]
                # V_str[2] = B[1,k1] -> V_full[1*8+k1]
                # V_str[3] = B[1,k2] -> V_full[1*8+k2]
                
                # W_str indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
                # Map to W_full: C[i,k] -> W_full[k*3+i]
                # W_str[0] = C[0,k1] -> W_full[k1*3+0]
                # W_str[1] = C[0,k2] -> W_full[k2*3+0]
                # W_str[2] = C[1,k1] -> W_full[k1*3+1]
                # W_str[3] = C[1,k2] -> W_full[k2*3+1]
                
                u_map = [0, 1, 3, 4]  # U_str index -> U_full index
                v_map = [0*8+k1, 0*8+k2, 1*8+k1, 1*8+k2]
                w_map = [k1*3+0, k2*3+0, k1*3+1, k2*3+1]
                
                for ui in range(4):
                    U[u_map[ui]][col + r] = U_str[ui][r]
                for vi in range(4):
                    V[v_map[vi]][col + r] = V_str[vi][r]
                for wi in range(4):
                    W[w_map[wi]][col + r] = W_str[wi][r]
            col += 7
        
        # A[0:2, 2] * B[2, :]
        for k in range(8):
            # C[0,k] += A[0,2] * B[2,k]
            U[0*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*3+0][col] = 1
            col += 1
            # C[1,k] += A[1,2] * B[2,k]
            U[1*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*3+1][col] = 1
            col += 1
        
        # A[2, 0:2] * B[0:2, :]
        for k in range(8):
            # C[2,k] += A[2,0] * B[0,k]
            U[2*3+0][col] = 1
            V[0*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
            # C[2,k] += A[2,1] * B[1,k]
            U[2*3+1][col] = 1
            V[1*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
        
        # A[2, 2] * B[2, :]
        for k in range(8):
            # C[2,k] += A[2,2] * B[2,k]
            U[2*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
        
        assert col == 68, f"Expected 68, got {col}"
        
        if check_decomposition(U, V, W, 68, 3, 3, 8):
            return U, V, W, 68
        else:
            return None, None, None, None
    
    # Try the hybrid decomposition
    U_h, V_h, W_h, rank_h = construct_hybrid_decomposition()
    if U_h is not None:
        if rank_h < best_rank:
            best_rank = rank_h
            best_result = {"U": U_h, "V": V_h, "W": W_h}
    
    # Try local search to improve further
    def local_search_improve(U, V, W, current_rank, n, m, p, max_iterations=50000):
        """Try to reduce the rank by zeroing out some columns."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        def compute_error(U, V, W):
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                        err += (total - 1.0)**2
            return err
        
        current_error = compute_error(U, V, W)
        
        # Try to zero out columns one by one
        for r in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            # Try zeroing column r
            U_save = [row[:] for row in U]
            V_save = [row[:] for row in V]
            W_save = [row[:] for row in W]
            
            for i in range(n*m):
                U[i][r] = 0
            for i in range(m*p):
                V[i][r] = 0
            for i in range(p*n):
                W[i][r] = 0
            
            # Check if still valid
            new_error = compute_error(U, V, W)
            if new_error < 1e-6:
                # Success, remove this column
                U = [row[:r] + row[r+1:] for row in U]
                V = [row[:r] + row[r+1:] for row in V]
                W = [row[:r] + row[r+1:] for row in W]
                current_rank -= 1
            else:
                # Restore
                U = U_save
                V = V_save
                W = W_save
        
        return U, V, W, current_rank
    
    if U_h is not None:
        U_improved, V_improved, W_improved, rank_improved = local_search_improve(
            U_h, V_h, W_h, 68, 3, 3, 8
        )
        if rank_improved < best_rank:
            best_rank = rank_improved
            best_result = {"U": U_improved, "V": V_improved, "W": W_improved}
    
    return best_result
# EVOLVE-BLOCK-END
