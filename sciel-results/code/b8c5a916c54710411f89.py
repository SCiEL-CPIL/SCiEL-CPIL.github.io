import numpy as np
import time
import itertools
from functools import lru_cache

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Target Tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U (n*m, R), V (m*p, R), W (p*n, R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # 1. Generate the known 88-rank solution as a safe fallback
    # This uses a Strassen-like decomposition for 2x2x2 blocks and schoolbook for the rest
    # Structure:
    # A is 3x4, B is 4x8.
    # We can view B as 4 blocks of 4x2.
    # Row 0 of A (1x4) * B (4x8) -> 1x8. Can be done with 4 mults (schoolbook) or less?
    # 1x4x4 matrix mult rank is 4 (schoolbook).
    # 2x4x4 matrix mult?
    # 3x4x4 matrix mult?
    
    # Let's stick to the provided 88-rank logic which is verified.
    def generate_strassen_hybrid():
        rank_total = 88
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        col = 0

        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ]

        def add_strassen(a_rb, a_cb, b_rb, b_cb, c_rb, c_cb):
            nonlocal col
            for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                for (ar, ac), coeff in a_coeffs.items():
                    U[(a_rb + ar) * m + (a_cb + ac), col] += coeff
                for (br, bc), coeff in b_coeffs.items():
                    V[(b_rb + br) * p + (b_cb + bc), col] += coeff
                for (cr, cc), coeff in c_coeffs.items():
                    W[(c_cb + cc) * n + (c_rb + cr), col] += coeff
                col += 1

        def add_1x2x2(a_rb, a_cb, b_rb, b_cb, c_rb, c_cb):
            nonlocal col
            terms = [
                ({(0,0):1}, {(0,0):1}, {(0,0):1}),
                ({(0,1):1}, {(1,0):1}, {(0,0):1}),
                ({(0,0):1}, {(0,1):1}, {(0,1):1}),
                ({(0,1):1}, {(1,1):1}, {(0,1):1}),
            ]
            for a_coeffs, b_coeffs, c_coeffs in terms:
                for (ar, ac), coeff in a_coeffs.items():
                    U[(a_rb + ar) * m + (a_cb + ac), col] += coeff
                for (br, bc), coeff in b_coeffs.items():
                    V[(b_rb + br) * p + (b_cb + bc), col] += coeff
                for (cr, cc), coeff in c_coeffs.items():
                    W[(c_cb + cc) * n + (c_rb + cr), col] += coeff
                col += 1

        # The 88 rank solution structure from the prompt's seed logic:
        # C11, C12, C13, C14 are computed using Strassen 2x2x2 blocks?
        # The seed program uses add_strassen for rows 0 and 1 of A interacting with 2x2 blocks of B?
        # Let's replicate the exact calls from the seed program to ensure 88 is valid.
        
        # C11 (A[0:2, 0:2] * B[0:2, 0:2] part? No, indices are global)
        # The seed program calls:
        # add_strassen(0, 0, 0, 0, 0, 0) -> A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
        # add_strassen(0, 2, 2, 0, 0, 0) -> A[0:2, 2:4], B[2:4, 0:2], C[0:2, 0:2]
        # This covers C[0:2, 0:2] using 14 terms? No, 7 terms per call.
        # Wait, Strassen 2x2x2 is 7 terms.
        # 4 calls for row 0 of C?
        # C[0, 0:2], C[0, 2:4], C[0, 4:6], C[0, 6:8]
        # The seed program calls add_strassen 4 times for row 0?
        # add_strassen(0, 0, 0, 0, 0, 0)
        # add_strassen(0, 2, 2, 0, 0, 0)
        # These two cover C[0:2, 0:2]? No.
        # Let's look at the indices.
        # add_strassen(a_rb, a_cb, b_rb, b_cb, c_rb, c_cb)
        # a_rb=0, a_cb=0: A rows 0,1 cols 0,1
        # b_rb=0, b_cb=0: B rows 0,1 cols 0,1
        # c_rb=0, c_cb=0: C rows 0,1 cols 0,1
        # This computes C[0:2, 0:2] from A[0:2, 0:2] and B[0:2, 0:2].
        # But we need C[0:2, 0:2] to be sum of A[0:2, 0:4] * B[0:4, 0:2].
        # So we need to sum contributions from B cols 0:2 and 2:4.
        # Call 1: A[0:2, 0:2] * B[0:2, 0:2] -> adds to C[0:2, 0:2]
        # Call 2: A[0:2, 2:4] * B[2:4, 0:2] -> adds to C[0:2, 0:2]
        # Total 14 terms for C[0:2, 0:2].
        
        # The seed program does:
        # C11 (C[0:2, 0:2]): 2 strassen calls (14 terms)
        # C12 (C[0:2, 2:4]): 2 strassen calls (14 terms)
        # C13 (C[0:2, 4:6]): 2 strassen calls (14 terms)
        # C14 (C[0:2, 6:8]): 2 strassen calls (14 terms)
        # Total for Row 0-1 of C: 56 terms.
        
        # C21 (C[2:3, 0:2]): add_1x2x2?
        # A[2:3, 0:2] (1x2) * B[0:2, 0:2] (2x2) -> C[2:3, 0:2] (1x2). Rank 4.
        # A[2:3, 2:4] (1x2) * B[2:4, 0:2] (2x2) -> C[2:3, 0:2] (1x2). Rank 4.
        # Total 8 terms.
        # Similarly for C22, C23, C24.
        # 4 * 8 = 32 terms.
        
        # Total 56 + 32 = 88 terms.
        
        add_strassen(0, 0, 0, 0, 0, 0)
        add_strassen(0, 2, 2, 0, 0, 0)
        # C12
        add_strassen(0, 0, 0, 2, 0, 2)
        add_strassen(0, 2, 2, 2, 0, 2)
        # C13
        add_strassen(0, 0, 0, 4, 0, 4)
        add_strassen(0, 2, 2, 4, 0, 4)
        # C14
        add_strassen(0, 0, 0, 6, 0, 6)
        add_strassen(0, 2, 2, 6, 0, 6)
        
        # C21
        add_1x2x2(2, 0, 0, 0, 2, 0)
        add_1x2x2(2, 2, 2, 0, 2, 0)
        # C22
        add_1x2x2(2, 0, 0, 2, 2, 2)
        add_1x2x2(2, 2, 2, 2, 2, 2)
        # C23
        add_1x2x2(2, 0, 0, 4, 2, 4)
        add_1x2x2(2, 2, 2, 4, 2, 4)
        # C24
        add_1x2x2(2, 0, 0, 6, 2, 6)
        add_1x2x2(2, 2, 2, 6, 2, 6)

        return U, V, W

    best_U, best_V, best_W = generate_strassen_hybrid()
    best_rank = 88

    # 2. Attempt to find a lower rank using Integer Search
    # Strategy: 
    # We want to find R < 88.
    # We will try to construct the decomposition using a pool of "atomic" rank-1 terms.
    # An atomic term is defined by vectors u (12,), v (32,), w (24,) with small integer entries.
    # However, the space of all such vectors is huge.
    # Instead, we use a "residual" approach.
    # We know the target tensor T is all 1s.
    # We can view the decomposition as filling a 3D grid.
    
    # A more robust structural search for small improvements (87, 86):
    # Check if the 88-rank solution has linear dependencies that can be exploited.
    # If we can find a linear combination of the 88 terms that equals 0, we might be able to swap one term for a simpler one or remove one if the rank is not minimal.
    # But tensor rank is not linear algebra rank of the matrix.
    
    # Let's try a targeted search for Rank 87.
    # If we can find a set of 87 terms that sum to T, we win.
    # We can use a SAT-like solver approach? Too slow.
    
    # Alternative: 
    # The 88-rank solution is composed of blocks.
    # Row 0-1 of C uses Strassen (14 terms per 2x2 block of C).
    # Row 2 of C uses Schoolbook (4 terms per 2x2 block of C).
    # Total 4 blocks in C horizontally (cols 0-1, 2-3, 4-5, 6-7).
    # Row 0-1: 4 * 14 = 56.
    # Row 2: 4 * 4 = 16? 
    # Wait, 1x2x2 is rank 4.
    # C[2, 0:2] needs A[2, 0:4] * B[0:4, 0:2].
    # A[2, 0:2] * B[0:2, 0:2] (1x2 * 2x2) -> 1x2. Rank 4.
    # A[2, 2:4] * B[2:4, 0:2] (1x2 * 2x2) -> 1x2. Rank 4.
    # Total 8 terms for C[2, 0:2].
    # So 4 blocks * 8 terms = 32 terms.
    # 56 + 32 = 88.
    
    # Can we do better for Row 2?
    # A[2, :] is a 1x4 vector. B[:, 0:2] is a 4x2 matrix.
    # We want to compute y = x * M where x is 1x4, M is 4x2.
    # This is equivalent to 2 separate dot products:
    # y0 = x0*M00 + x1*M10 + x2*M20 + x3*M30
    # y1 = x0*M01 + x1*M11 + x2*M21 + x3*M31
    # Can we compute this with fewer than 8 multiplications?
    # The rank of the 1x4x2 multiplication tensor is 4? 
    # No, 1x4x2 tensor T[i,j,k] = 1 if i=0.
    # T is 1x4x2.
    # Rank of 1xn x nxm?
    # For 1x4x2, the rank is 4. (You need 4 mults to compute 2 outputs? No, each output is a dot product of length 4. They share inputs x.
    # Can we share?
    # y0 = sum x_i M_i0
    # y1 = sum x_i M_i1
    # This is just matrix multiplication.
    # Is there a rank 3 decomposition?
    # If rank is 4, we are already optimal for the 1x4x2 block.
    # So 8 terms for the 1x4x4 block (two 1x4x2s) is optimal?
    # Wait, the block is 1x4 (A) * 4x4 (B) -> 1x4 (C).
    # This is 1x4x4.
    # Rank of 1xn x nxn is n.
    # So 1x4x4 has rank 4.
    # My previous calculation:
    # C[2, 0:2] comes from A[2, 0:4] * B[0:4, 0:2].
    # This is a 1x4x2 tensor. Rank 4.
    # C[2, 2:4] comes from A[2, 0:4] * B[0:4, 2:4].
    # This is a 1x4x2 tensor. Rank 4.
    # Do they share factors?
    # A[2, 0:4] is shared.
    # The decomposition of 1x4x2 uses x (1x4) and M (4x2).
    # If we decompose 1x4x4, we use x (1x4) and M (4x4).
    # Rank of 1x4x4 is 4.
    # So we can compute the entire row 2 of C (1x8) with 4 multiplications?
    # No, 1x4x4 rank is 4.
    # But we have 1x4x8.
    # Rank of 1x4x8 is 4?
    # T[i,j,k] = 1.
    # y_k = sum_j x_j M_jk.
    # This is a linear map from M to y for fixed x.
    # The tensor is separable?
    # T = sum_r u_r v_r w_r.
    # For 1xn x nxp, the rank is n.
    # So 1x4x8 has rank 4.
    # 
    # If this is true, we can compute Row 2 of C using only 4 terms!
    # Currently we use 32 terms (4 blocks of 8).
    # If we use 4 terms for the whole row, we save 28 terms.
    # New Rank = 56 (Row 0-1) + 4 (Row 2) = 60.
    
    # Let's verify the rank of 1x4x8.
    # We want C[2, k] = sum_{j=0}^3 A[2, j] B[j, k].
    # This is a dot product of A[2,:] and B[:,k].
    # We can write this as:
    # C[2, k] = sum_{r=0}^3 U[2, r] V[r, k] ? No, V is indexed by j,k.
    # In the tensor decomposition:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For i=2, T[2,j,k] = 1.
    # We need sum_r U[2,j,r] V[j,k,r] W[k,2,r] = 1.
    
    # If we set W[k,2,r] = 1 for all k,r?
    # Then sum_r U[2,j,r] V[j,k,r] = 1.
    # This requires the matrix M_{jk} = sum_r U[2,j,r] V[j,k,r] to be all 1s.
    # M is 4x8.
    # We need to factor a 4x8 matrix of all 1s into sum of rank-1 matrices.
    # The rank of a 4x8 matrix of all 1s is 1.
    # So we can do it with R=1?
    # If R=1, U[2,j,0] * V[j,k,0] = 1.
    # Let U[2,j,0] = 1, V[j,k,0] = 1.
    # Then W[k,2,0] must be 1.
    # So yes, we can compute the entire row 2 with 1 term?
    # Wait.
    # The term contributes to ALL i, j, k.
    # We are not just computing row 2. We are computing the whole tensor.
    # The factors U, V, W are global.
    # We cannot isolate row 2 without affecting rows 0 and 1.
    
    # However, we CAN design the decomposition such that the terms used for Row 2 do not contribute to Row 0 and 1.
    # This requires U[0,j,r] = 0 and U[1,j,r] = 0 for the terms used for Row 2.
    # If we have 4 terms dedicated to Row 2, we can set U[0,j,r]=0, U[1,j,r]=0 for those r.
    # Then they don't interfere.
    
    # So, can we compute Row 2 (1x4x8) with 4 terms?
    # Yes, rank of 1x4x8 is 4.
    # So we can replace the 32 terms for Row 2 with 4 terms.
    # Total Rank = 56 + 4 = 60.
    
    # Let's construct this 60-rank solution.
    
    def generate_60_rank():
        R = 60
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        
        # Part 1: Rows 0-1 of C (56 terms)
        # Same as before
        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ]

        def add_strassen(a_rb, a_cb, b_rb, b_cb, c_rb, c_cb):
            nonlocal col
            for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                for (ar, ac), coeff in a_coeffs.items():
                    U[(a_rb + ar) * m + (a_cb + ac), col] += coeff
                for (br, bc), coeff in b_coeffs.items():
                    V[(b_rb + br) * p + (b_cb + bc), col] += coeff
                for (cr, cc), coeff in c_coeffs.items():
                    W[(c_cb + cc) * n + (c_rb + cr), col] += coeff
                col += 1

        add_strassen(0, 0, 0, 0, 0, 0)
        add_strassen(0, 2, 2, 0, 0, 0)
        add_strassen(0, 0, 0, 2, 0, 2)
        add_strassen(0, 2, 2, 2, 0, 2)
        add_strassen(0, 0, 0, 4, 0, 4)
        add_strassen(0, 2, 2, 4, 0, 4)
        add_strassen(0, 0, 0, 6, 0, 6)
        add_strassen(0, 2, 2, 6, 0, 6)
        
        # Part 2: Row 2 of C (4 terms)
        # We need to compute C[2, k] = sum_j A[2, j] B[j, k]
        # Using 4 terms r=0..3.
        # We want sum_r U[2, j, r] V[j, k, r] W[k, 2, r] = 1 * A[2,j] * B[j,k] ? No.
        # The tensor T is 1.
        # We need sum_r U[2, j, r] V[j, k, r] W[k, 2, r] = 1.
        # We can choose:
        # U[2, j, r] = 1 for all j, r?
        # V[j, k, r] = B[j, k] ? No, V is constant.
        
        # The decomposition must be independent of A and B values.
        # The formula is C = A B.
        # The tensor T encodes the structure.
        # T[i,j,k] = 1.
        # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
        
        # For i=2:
        # sum_r U[2,j,r] V[j,k,r] W[k,2,r] = 1.
        
        # We can set W[k,2,r] = 1 for all k,r.
        # Then sum_r U[2,j,r] V[j,k,r] = 1.
        # We need a 4x8 matrix M_{jk} = sum_r U[2,j,r] V[j,k,r] to be all 1s.
        # We can factor the all-ones 4x8 matrix into 4 rank-1 matrices.
        # M = sum_{r=0}^3 u_r v_r^T.
        # Let u_r be a 4-vector and v_r be an 8-vector.
        # We can simply use the columns of the all-ones matrix?
        # Rank of all-ones 4x8 is 1.
        # So we can do it with 1 term?
        # If we use 1 term:
        # U[2,j,0] = 1, V[j,k,0] = 1, W[k,2,0] = 1.
        # Then sum = 1.
        # But we need to ensure this term doesn't interfere with i=0,1.
        # For i=0,1, U[i,j,0] should be 0.
        # And W[k,0,0], W[k,1,0] should be 0.
        
        # So we can add 1 term for the entire Row 2!
        # Total Rank = 56 + 1 = 57.
        
        # Let's verify.
        # Term 56:
        # U[2, j, 56] = 1 for all j in 0..3.
        # V[j, k, 56] = 1 for all j in 0..3, k in 0..7.
        # W[k, 2, 56] = 1 for all k in 0..7.
        # W[k, 0, 56] = 0.
        # W[k, 1, 56] = 0.
        # U[0, j, 56] = 0.
        # U[1, j, 56] = 0.
        
        # Contribution to T[0, j, k]:
        # sum_r ... U[0,j,r] ... = 0 (since U[0,j,56]=0 and other terms don't touch i=2? 
        # Wait, the Strassen terms touch i=0,1. Do they touch i=2?
        # The Strassen calls were for a_rb=0,1. So U indices are (0*m+j) and (1*m+j).
        # U[2*m+j, r] is untouched by Strassen terms.
        # So U[2,j,r] = 0 for r < 56.
        
        # So for i=0,1, the new term contributes 0.
        # For i=2, the new term contributes 1*1*1 = 1.
        # The Strassen terms contribute 0 to i=2.
        # So T[2,j,k] = 1.
        
        # This works!
        
        U[2 * m + 0, col] = 1
        U[2 * m + 1, col] = 1
        U[2 * m + 2, col] = 1
        U[2 * m + 3, col] = 1
        
        for j in range(m):
            for k in range(p):
                V[j * p + k, col] = 1
                W[k * n + 2, col] = 1
        col += 1
        
        return U, V, W

    def verify_decomposition(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try the 57-rank solution
    U_57, V_57, W_57 = generate_60_rank()
    if verify_decomposition(U_57, V_57, W_57):
        best_U, best_V, best_W = U_57, V_57, W_57
        best_rank = 57

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

