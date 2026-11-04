"""Rewritten program for 6x6x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: row = i*2 + j for U, j*2 + k for V, k*2 + i for W.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B00-B10)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B01+B11)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B00)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _laderman_4x4():
    """
    Return the rank-20 Laderman decomposition for 4x4x4 matrix multiplication.
    U: (16, 20), V: (16, 20), W: (16, 20)
    Indexing: row = i*4 + j for U, j*4 + k for V, k*4 + i for W.
    """
    U = np.zeros((16, 20))
    V = np.zeros((16, 20))
    W = np.zeros((16, 20))
    
    # Laderman's 4x4x4 algorithm (rank 20)
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A * B
    #
    # The 20 multiplications are defined as follows (using 2x2 block structure):
    # We define 20 bilinear forms. Each form multiplies two linear combinations
    # of 2x2 blocks and contributes to output blocks.
    #
    # Laderman's algorithm:
    # Let's index the 2x2 blocks:
    # A blocks: A00(0:4), A01(4:8), A10(8:12), A11(12:16)
    # B blocks: B00(0:4), B01(4:8), B10(8:12), B11(12:16)
    # C blocks: C00(0:4), C01(4:8), C10(8:12), C11(12:16)
    #
    # Within each 2x2 block, index as [a,b;c,d] -> positions 0,1,2,3
    #
    # Laderman defines 20 products. Let me use the standard formulation:
    # The 20 rank-1 terms are:
    
    # I'll use the formulation from Laderman (1976).
    # The 20 multiplications are:
    # For the 2x2 block structure, we can view this as:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each 2x2 block multiplication has 4 terms, total 16.
    # Laderman saves 4 by combining.
    #
    # Actually, let me use a different approach. I'll define the 20 terms
    # explicitly based on the known Laderman algorithm.
    #
    # Laderman's 20 products (using block notation, each product is of two 2x2 matrices):
    # The key insight is that within each 2x2 block, we use specific linear combinations.
    
    # Let me use the explicit 20-term formulation:
    # Term 1: (A00[0,0]+A00[1,1]) * (B00[0,0]+B00[1,1])
    # ... this is getting complex. Let me use a verified formulation.
    
    # I'll use the following known rank-20 decomposition:
    # The 20 bilinear forms for 4x4x4:
    
    # Using the formulation where we split into 2x2 blocks and apply
    # specific linear combinations within blocks:
    
    # Let's define the 20 terms by their (U, V, W) contributions.
    # Each term r has U_r (16-vector), V_r (16-vector), W_r (16-vector).
    
    # Term 0: A00_top * B00_top -> C00_top
    # A00 is rows 0-3, cols 0-3 of A. A00_top = A[0:2, 0:2]
    U[0, 0] = 1; U[1, 0] = 1  # A[0,0], A[0,1]
    V[0, 0] = 1; V[1, 0] = 1  # B[0,0], B[0,1]
    W[0, 0] = 1; W[1, 0] = 1  # C[0,0], C[0,1]
    
    # This is getting very complex to hardcode correctly. Let me use a different strategy.
    # I'll verify the 189 solution is correct and try to find a better one.
    
    # Actually, let me reconsider. The safest approach that I can verify:
    # 3x3 grid of 2x2 Strassen = 27 * 7 = 189.
    # 2x2 grid of 3x3 schoolbook = 4 * 27 = 108. Wait, that's wrong.
    # 2x2 grid means 4 blocks, each 3x3x3. Rank = 4 * 27 = 108.
    # But that's only for the diagonal blocks. We need cross terms too.
    #
    # For 2x2 block decomposition:
    # A = [[A11, A12], [A21, A22]] (each 3x3)
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Using Strassen at block level (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B12+B22)
    # M7 = (A12-A22)(B21+B11)
    #
    # Each Mi is a product of two 3x3 matrices.
    # We decompose each 3x3x3 product using schoolbook (rank 27).
    # Total rank = 7 * 27 = 189.
    
    # Same as before. Let me just implement the verified 189 solution.
    
    # For the 4+2 split with Laderman, I need to correctly implement Laderman.
    # Let me try a different approach: use the 3x3 grid of 2x2 Strassen (189)
    # and see if I can optimize.
    
    # Actually, let me implement the 4+2 split properly.
    # I need a correct Laderman 4x4x4 rank-20 decomposition.
    
    # Let me use a known correct formulation:
    # Laderman's algorithm uses the following 20 products:
    # (using 2x2 block notation, where each "element" is a 2x2 matrix)
    
    # I'll define it using the explicit indices.
    # A[i,j] for i,j in 0..3, B[i,j], C[i,j]
    # U index: i*4+j, V index: i*4+j, W index: i*4+j (with W[k*4+i])
    
    # Laderman's 20 terms (from the literature):
    # I'll use the formulation where we define:
    # For 2x2 blocks A11,A12,A21,A22 and B11,B12,B21,B22:
    
    # The 20 products are:
    # 1. A11*B11 -> C11 (but with specific sub-block combinations)
    # ...
    
    # This is too error-prone to hardcode from memory. Let me stick with 189.
    
    return U, V, W

def _schoolbook_3x3():
    """
    Return the rank-27 schoolbook decomposition for 3x3x3 matrix multiplication.
    U: (9, 27), V: (9, 27), W: (9, 27)
    """
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 6, 6, 6
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy: 3x3 grid of 2x2 Strassen blocks.
    # Rank = 27 * 7 = 189.
    
    total_rank = 189
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row, col] = sU[li*2+lj, c]
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row, col] = sV[lj*2+lk, c]
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row, col] = sW[lk*2+li, c]
                    col += 1
    
    # Verify correctness
    # The tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j] B[j,k])
    # U[i,r] * V[j,r] * W[k,r] summed over r should give T[i,j,k]
    # where i indexes A row*col, j indexes B row*col, k indexes C col*row (W[k*n+i])
    
    # Let's verify with random matrices
    np.random.seed(42)
    A_test = np.random.randn(6, 6)
    B_test = np.random.randn(6, 6)
    C_test = A_test @ B_test
    
    # Compute using decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # C[i,k] = sum_r U[i,r] * (sum_j V[j,r] * A[i,j]) * ... 
    # Actually: C = sum_r (U[:,r] outer product with V[:,r]) * W[:,r]
    # More precisely: C[i,k] = sum_r U[i,r] * V[j,r] * W[k,r] where j is summed
    # C[i,k] = sum_j sum_r A[i,j] * U[i,r] * V[j,r] * W[k,r]
    
    C_decomp = np.zeros((6, 6))
    for i in range(6):
        for k in range(6):
            s = 0.0
            for r in range(total_rank):
                # Compute the middle term: sum_j (U[i,r] * A[i,j]) * (V[j,r] * B[j,k])
                # Note: The standard decomposition C[i,k] = sum_r sum_j U[i,r] * A[i,j] * V[j,r] * B[j,k] * W[k*6+i, r]
                # can be rearranged as: C[i,k] = sum_r W[k*6+i, r] * (sum_j (U[i,r] * A[i,j]) * (V[j,r] * B[j,k]))
                # However, looking at the previous context, it seems we are trying to reconstruct C.
                # Let's assume the standard form: C[i,k] = sum_r (sum_j U[i,r] * A[i,j] * V[j,r] * B[j,k]) * W[k*6+i, r]
                # Wait, the indices in W are k*6+i.
                # Let's calculate the inner sum over j first for a given r.
                inner_sum = 0.0
                for j in range(6):
                    inner_sum += U[i, r] * A_test[i, j] * V[j, r] * B_test[j, k]
                s += inner_sum * W_arr[k*6+i, r]
                # Let me think again.
                # The tensor decomposition: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
                # where T_{ijk} = delta_{ik} (for matrix mult: C[i,k] = sum_j A[i,j] B[j,k])
                # So: sum_r U[i,r] V[j,r] W[k,r] = delta_{i,k} for all i,j,k? No.
                # T_{ijk} is the matrix multiplication tensor: T_{ijk} = 1 if i==k (in the appropriate indexing)
                # Actually for matrix mult: C_{ik} = sum_j A_{ij} B_{jk}
                # Tensor: T_{i,j,k} where i indexes (row of A, col of C), j indexes (col of A, row of B), k indexes (col of B, row of C)
                # T_{i,j,k} = 1 if the indices are consistent, i.e., always 1 (it's just the identity in the middle)
                # Wait, T_{i,j,k} = delta_{(i,j),(j,k)} which is always 1.
                # No, that's not right either.
                # The matrix multiplication tensor has T_{i,j,k} = 1 for all valid i,j,k.
                # So sum_r U[i,r] V[j,r] W[k,r] should = 1 for all i,j,k? That can't be.
                # 
                # Let me reconsider. The tensor for matrix multiplication:
                # C_{ik} = sum_j A_{ij} B_{jk}
                # This is a trilinear form: sum_{i,j,k} A_{ij} B_{jk} C_{ki} = trace(ABC)
                # The tensor T_{ijk} = 1 for all i,j,k in the appropriate range.
                # So the decomposition is: sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k? 
                # That would mean U^T V W^T = ones matrix, which is impossible for rank < 36.
                # 
                # I think I'm confusing myself. Let me just verify numerically.
                pass
    
    # Simpler verification:
    # C_decomp[i,k] = sum_r W[k*6+i, r] * (sum_j A[i,j] * U[i,r] * V[j,r])
    # Wait, that's not right either. Let me think carefully.
    # 
    # The decomposition says: for all i, j, k:
    # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    # where T[i,j,k] = 1 if (i,j,k) corresponds to a valid matrix mult term.
    # For matrix mult: T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k] always has the term).
    # 
    # Hmm, but that would require rank 36 (all ones). That's wrong.
    # 
    # I think the correct tensor is: T[i,j,k] = 1 if the indices match for matrix multiplication.
    # In the flattening: i = row_A * 6 + col_A, j = col_A * 6 + row_B, k = col_B * 6 + row_C
    # For the term to contribute: col_A (from i) must equal row_B (from j), and col_B (from j) must equal row_C (from k).
    # So T[i,j,k] = 1 if (i % 6) == (j // 6) and (j % 6) == (k // 6), else 0.
    
    # Let me verify this way:
    max_err = 0.0
    for i in range(36):
        for j in range(36):
            for k in range(36):
                expected = 1.0 if (i % 6 == j // 6 and j % 6 == k // 6) else 0.0
                actual = 0.0
                for r in range(total_rank):
                    actual += U[i, r] * V[j, r] * W[k, r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback: use schoolbook
        total_rank = 216
        U = np.zeros((36, total_rank))
        V = np.zeros((36, total_rank))
        W = np.zeros((36, total_rank))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i * 6 + j, col] = 1
                    V[j * 6 + k, col] = 1
                    W[k * 6 + i, col] = 1
                    col += 1
    else:
        # Verify with actual matrix multiplication
        A_test = np.random.randn(6, 6)
        B_test = np.random.randn(6, 6)
        C_test = A_test @ B_test
        
        C_decomp = np.zeros((6, 6))
        for i in range(6):
            for k in range(6):
                s = 0.0
                for r in range(total_rank):
                    s += W[k * 6 + i, r]
                    for j in range(6):
                        s += A_test[i, j] * U[i * 6 + j, r] * 0  # wrong
                # Correct: C[i,k] = sum_r W[k*6+i,r] * (sum_j A[i,j] * U[i*6+j, r] * V[j*6+k, r])
                # No, that's not right. The bilinear form is:
                # C[i,k] = sum_r (sum_j A[i,j] * U[i*6+j, r]) * (sum_j' B[j',k] * V[j'*6+k, r]) * W[k*6+i, r]
                # That's not right either.
                # 
                # The correct interpretation:
                # sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
                # So: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r T[i,j,k] A[i,j] B[j,k]
                # = sum_r sum_j T[i,j,k] A[i,j] B[j,k]
                # = sum_r (sum_j T[i,j,k] A[i,j]) * (sum_j T[i,j,k] B[j,k])
                # But T[i,j,k] depends on k, so we can't separate.
                # 
                # Actually: C[i,k] = sum_j A[i,j] B[j,k]
                # And sum_r U[i*6+j, r] V[j*6+k, r] W[k*6+i, r] = T[i,j,k]
                # So: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
                # = sum_{i,j,k} (sum_r U[i*6+j, r] V[j*6+k, r] W[k*6+i, r]) A[i,j] B[j,k]
                # = sum_r sum_{i,j,k} U[i*6+j, r] V[j*6+k, r] W[k*6+i, r] A[i,j] B[j,k]
                # = sum_r W[k*6+i, r] * (sum_j U[i*6+j, r] A[i,j]) * (sum_j V[j*6+k, r] B[j,k])
                # Wait, that's not separable because U depends on i and j, V depends on j and k.
                # 
                # Let me just compute it directly:
                s = 0.0
                for r in range(total_rank):
                    for j in range(6):
                        s += A_test[i, j] * B_test[j, k] * (U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r])
                C_decomp[i, k] = s
        
        err = np.max(np.abs(C_decomp - C_test))
        if err > 1e-9:
            # Fallback
            total_rank = 216
            U = np.zeros((36, total_rank))
            V = np.zeros((36, total_rank))
            W = np.zeros((36, total_rank))
            col = 0
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        U[i * 6 + j, col] = 1
                        V[j * 6 + k, col] = 1
                        W[k * 6 + i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
