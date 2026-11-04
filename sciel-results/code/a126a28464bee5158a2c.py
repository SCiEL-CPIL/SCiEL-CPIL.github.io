import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We will construct the decomposition by splitting the output dimension p.
    # Let p = p1 + p2.
    # We can compute C[:, :p1] using a decomposition of size n x m x p1.
    # We can compute C[:, p1:] using a decomposition of size n x m x p2.
    # The total rank is rank(n, m, p1) + rank(n, m, p2).
    
    # For n=4, m=4, p=7:
    # Option A: p1=4, p2=3.
    # Rank(4,4,4) can be 49 (Strassen).
    # Rank(4,4,3) can be 48 (Schoolbook) or lower?
    # 4x4x3 can be seen as 2x(2x4x3).
    # 2x4x3 is 2x2x3 + 2x2x3 + 2x2x3 + 2x2x3? No.
    # 4x4x3 = 4 * (1x4x3)? No.
    # 4x4x3 = 2 * (2x4x3).
    # 2x4x3 = 2x2x3 + 2x2x3.
    # Rank(2,2,3) is known to be 7? No, 2x2x2 is 7. 2x2x3 is at least 7.
    # Actually, Rank(2,2,3) = 7 is not correct. Rank(2,2,2)=7.
    # Rank(2,2,3) is 7? No, it's larger.
    # Schoolbook for 4x4x3 is 48.
    # Is there a better decomposition for 4x4x3?
    # We can use Strassen for the 4x4x4 part (rank 49).
    # For the 4x4x3 part, we can just use schoolbook (rank 48).
    # Total rank = 49 + 48 = 97.
    
    # Let's implement a helper to generate the U, V, W matrices for a given n, m, p
    # using either Schoolbook or Strassen (if n=m=p is a power of 2 and we choose Strassen).
    
    def schoolbook_matrices(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_matrices(n, m, p):
        # Only valid if n == m == p and n is a power of 2.
        # We will build it recursively.
        # Base case: 1x1x1
        if n == 1:
            return [[1]], [[1]], [[1]]
        
        half = n // 2
        # Split A, B, C into 4 blocks of size half x half.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # We need to construct U, V, W for the n x n x n tensor.
        # The tensor T_n is the sum of 7 tensors, each corresponding to one M.
        # Each M is a product of two linear combinations of blocks.
        
        # Let's represent the linear combinations.
        # For each of the 7 multiplications, we have:
        # U_part: combination of A blocks
        # V_part: combination of B blocks
        # The result is added to C blocks.
        
        # We will build the global U, V, W by iterating over the 7 Strassen steps.
        # For each step, we recursively generate the decomposition for the half-size matrices.
        
        # Note: The recursive call returns U_rec, V_rec, W_rec for size half.
        # We need to embed these into the size n matrices.
        
        # Let's define the 7 steps as tuples of:
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (block_index, sign) for A
        # V_coeffs: list of (block_index, sign) for B
        # W_coeffs: list of (block_index, sign) for C
        
        # Block indices for n x n matrix split into 2x2 blocks of size half:
        # 0: (0, 0) -> A11
        # 1: (0, 1) -> A12
        # 2: (1, 0) -> A21
        # 3: (1, 1) -> A22
        
        # The 7 steps:
        # 1: U: (0,1), (3,1); V: (0,1), (3,1); W: (0,1), (3,-1), (1,1), (2,-1) ? 
        #    C11 = M1 + M4 - M5 + M7
        #    C12 = M3 + M5
        #    C21 = M2 + M4
        #    C22 = M1 - M2 + M3 + M6
        
        # Let's list the contributions of each M to C blocks:
        # M1: C11(+1), C22(+1)
        # M2: C21(+1), C22(-1)
        # M3: C12(+1), C22(+1)
        # M4: C11(+1), C21(+1)
        # M5: C11(-1), C12(+1)
        # M6: C22(+1)
        # M7: C11(+1)
        
        # And the inputs:
        # M1: A11+A22, B11+B22
        # M2: A21+A22, B11
        # M3: A11, B12-B22
        # M4: A22, B21-B11
        # M5: A11+A12, B22
        # M6: A21-A11, B11+B12
        # M7: A12-A22, B21+B22
        
        # We will create a list of 7 dictionaries defining the linear combinations.
        steps = [
            {
                'U': [(0, 1), (3, 1)],
                'V': [(0, 1), (3, 1)],
                'W': [(0, 1), (3, 1)] # C11, C22
            },
            {
                'U': [(2, 1), (3, 1)],
                'V': [(0, 1)],
                'W': [(2, 1), (3, -1)] # C21, C22
            },
            {
                'U': [(0, 1)],
                'V': [(1, 1), (3, -1)],
                'W': [(1, 1), (3, 1)] # C12, C22
            },
            {
                'U': [(3, 1)],
                'V': [(2, 1), (0, -1)],
                'W': [(0, 1), (2, 1)] # C11, C21
            },
            {
                'U': [(0, 1), (1, 1)],
                'V': [(3, 1)],
                'W': [(0, -1), (1, 1)] # C11, C12
            },
            {
                'U': [(2, 1), (0, -1)],
                'V': [(0, 1), (1, 1)],
                'W': [(3, 1)] # C22
            },
            {
                'U': [(1, 1), (3, -1)],
                'V': [(2, 1), (3, 1)],
                'W': [(0, 1)] # C11
            }
        ]
        
        # Recursive call for half size
        U_rec, V_rec, W_rec = strassen_matrices(half, half, half)
        rank_rec = len(U_rec[0])
        
        # Total rank for this level is 7 * rank_rec
        total_rank = 7 * rank_rec
        
        U = [[0] * total_rank for _ in range(n * n)]
        V = [[0] * total_rank for _ in range(n * n)]
        W = [[0] * total_rank for _ in range(n * n)]
        
        # Helper to add a block of the recursive matrices to the global matrices
        def add_block(global_mat, block_row_offset, block_col_offset, rec_mat, signs, rank_offset):
            # global_mat is n*n x total_rank
            # rec_mat is half*half x rank_rec
            # signs is a list of (block_idx, sign) for the row/col dimension
            # Actually, the signs apply to the linear combination of the blocks.
            # For U, the rows are A indices. A is n x n.
            # A[i, j] is in block (i//half, j//half).
            # The linear combination is sum_sign * A_block.
            # So U_global[i, j, r] = sum_{(b, s) in U_coeffs} s * U_rec[i%half, j%half, r-rec_rank_offset]
            
            pass
            
        # This embedding is tricky. Let's think about the indices.
        # U_global has rows indexed by (i, j) where 0<=i,j<n.
        # For a specific Strassen step, say step s, we have coefficients for A blocks.
        # Let the coefficients be c_b for block b.
        # Then U_global[i, j, r] = c_{block(i,j)} * U_rec[i%half, j%half, r_local]
        # where r = s * rank_rec + r_local.
        
        # Similarly for V (indexed by j, k) and W (indexed by k, i).
        # Note: The recursive V is for B (half x half).
        # The recursive W is for C (half x half).
        
        for s in range(7):
            u_coeffs = steps[s]['U']
            v_coeffs = steps[s]['V']
            w_coeffs = steps[s]['W']
            
            # Map block index to coefficient
            # Block index is 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            u_coeff_map = {b: 0 for b in range(4)}
            for b, sign in u_coeffs:
                u_coeff_map[b] += sign
                
            v_coeff_map = {b: 0 for b in range(4)}
            for b, sign in v_coeffs:
                v_coeff_map[b] += sign
                
            w_coeff_map = {b: 0 for b in range(4)}
            for b, sign in w_coeffs:
                w_coeff_map[b] += sign

            # Now fill the global matrices for this step
            for i in range(n):
                for j in range(n):
                    # Determine block for A[i, j]
                    bi = i // half
                    bj = j // half
                    block_idx = bi * 2 + bj
                    coeff = u_coeff_map[block_idx]
                    if coeff != 0:
                        # Local indices in recursive matrix
                        li = i % half
                        lj = j % half
                        # Add to global U
                        for r_local in range(rank_rec):
                            r_global = s * rank_rec + r_local
                            U[i * n + j][r_global] += coeff * U_rec[li * half + lj][r_local]
                            
            for j in range(n):
                for k in range(n):
                    # Determine block for B[j, k]
                    bj = j // half
                    bk = k // half
                    block_idx = bj * 2 + bk
                    coeff = v_coeff_map[block_idx]
                    if coeff != 0:
                        lj = j % half
                        lk = k % half
                        for r_local in range(rank_rec):
                            r_global = s * rank_rec + r_local
                            V[j * n + k][r_global] += coeff * V_rec[lj * half + lk][r_local]
                            
            for k in range(n):
                for i in range(n):
                    # Determine block for C[i, k]
                    # Note: W index in global is k*n + i.
                    # C is n x n. C[i, k].
                    # Block index for C[i, k] is (i//half, k//half).
                    ci = i // half
                    ck = k // half
                    block_idx = ci * 2 + ck
                    coeff = w_coeff_map[block_idx]
                    if coeff != 0:
                        li = i % half
                        lk = k % half
                        for r_local in range(rank_rec):
                            r_global = s * rank_rec + r_local
                            W[k * n + i][r_global] += coeff * W_rec[li * half + lk][r_local]
                            
        return U, V, W

    # Function to combine decompositions for p = p1 + p2
    def combine_decompositions(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        total_rank = rank1 + rank2
        
        # U has shape (n*m, total_rank)
        # For the first p1 columns of B, we use V1.
        # V1 has shape (m*p1, rank1).
        # V2 has shape (m*p2, rank2).
        # We need to construct V_global of shape (m*p, total_rank).
        # V_global[j, k, r]
        # If k < p1, it's part of B1. V_global[j, k, r] = V1[j, k, r] for r < rank1.
        # If k >= p1, it's part of B2. V_global[j, k, r] = V2[j, k-p1, r-rank1] for r >= rank1.
        
        U_global = [[0] * total_rank for _ in range(n * m)]
        V_global = [[0] * total_rank for _ in range(m * p)]
        W_global = [[0] * total_rank for _ in range(p * n)]
        
        # Fill U_global
        # U is the same for both parts?
        # Yes, A is the same.
        # U_global[i, j, r] = U1[i, j, r] for r < rank1
        # U_global[i, j, r] = U2[i, j, r-rank1] for r >= rank1
        for i in range(n):
            for j in range(m):
                for r in range(rank1):
                    U_global[i * m + j][r] = U1[i * m + j][r]
                for r in range(rank2):
                    U_global[i * m + j][rank1 + r] = U2[i * m + j][r]
                    
        # Fill V_global
        for j in range(m):
            for k in range(p):
                if k < p1:
                    for r in range(rank1):
                        V_global[j * p + k][r] = V1[j * p1 + k][r]
                    # V2 part is 0
                else:
                    # V1 part is 0
                    k2 = k - p1
                    for r in range(rank2):
                        V_global[j * p + k][rank1 + r] = V2[j * p2 + k2][r]
                        
        # Fill W_global
        # W index is k*n + i.
        # W1 corresponds to C[:, :p1]. So W1[k*n+i] is for k < p1.
        # W2 corresponds to C[:, p1:]. So W2[k2*n+i] is for k >= p1.
        for k in range(p):
            for i in range(n):
                if k < p1:
                    for r in range(rank1):
                        W_global[k * n + i][r] = W1[k * n + i][r]
                else:
                    k2 = k - p1
                    for r in range(rank2):
                        W_global[k * n + i][rank1 + r] = W2[k2 * n + i][r]
                        
        return U_global, V_global, W_global

    # Strategy:
    # Split p=7 into p1=4 and p2=3.
    # For n=4, m=4, p1=4: Use Strassen (Rank 49).
    # For n=4, m=4, p2=3: Use Schoolbook (Rank 48).
    
    U1, V1, W1 = strassen_matrices(4, 4, 4)
    U2, V2, W2 = schoolbook_matrices(4, 4, 3)
    
    U, V, W = combine_decompositions(4, 4, 4, 3, U1, V1, W1, U2, V2, W2)
    
    # Verification step (optional but good for safety)
    # def verify(U, V, W, n, m, p):
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 val = 0
    #                 for r in range(len(U[0])):
    #                     val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    #                 if val != 1:
    #                     return False
    #     return True
    # 
    # if not verify(U, V, W, n, m, p):
    #     # Fallback to schoolbook
    #     U, V, W = schoolbook_matrices(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
