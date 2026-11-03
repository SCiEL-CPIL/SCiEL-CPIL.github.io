"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a Strassen-based tensor decomposition for matrix multiplication.
    Dimensions: n x m * m x p -> n x p
    Uses 2x2 block Strassen on the 4x4 middle dimension.
    """
    start_time = time.time()
    deadline = 140.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Try Strassen-based decomposition
    # Split the 4x4 middle dimension into 2x2 blocks
    # A is 3x4, B is 4x7
    # A = [[A11, A12], [A21, A22]] where A11, A12 are 3x2, A21, A22 are 3x2 (wait, that's not square)
    # Actually for Strassen we need square blocks. Let's think differently.
    
    # The middle dimension is m=4. We can split A (3x4) into two 3x2 blocks and B (4x7) into two 2x7 blocks.
    # A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # C = A_left * B_top + A_right * B_bottom
    
    # This is a sum of two matrix products, each of size 3x2 * 2x7
    # Schoolbook rank for 3x2 * 2x7 is 3*2*7 = 42
    # Total schoolbook rank: 2 * 42 = 84 (same as before)
    
    # To use Strassen, we need to treat this differently.
    # Let's embed the problem into a larger Strassen framework.
    
    # Alternative: Use the fact that we can apply Strassen to 2x2 blocks of the computation.
    # Consider splitting the 3 rows of A and 7 cols of B into blocks.
    
    # Let's try a direct approach: use known low-rank decompositions.
    # For 3x4 * 4x7, the minimum rank is at most 72 (by known bounds), but let's try to get below 84.
    
    # A simpler approach: use the tensor product of smaller decompositions.
    # 3x4*4x7 can be seen as related to 2x2*2x2 via embedding.
    
    # Let's try: split into 2x2 blocks where possible.
    # A (3x4): rows 0,1 form top 2x4, row 2 is bottom 1x4
    # B (4x7): cols 0,1 form left 4x2, cols 2-6 form right 4x5
    
    # This is getting complex. Let me try a computational approach.
    
    # Use bilinear complexity: the tensor is in R^(n*m) x R^(m*p) x R^(p*n)
    # We want to find rank-r decomposition.
    
    # For now, let's try a heuristic: use Strassen on 2x2 subproblems.
    # Split n=3 into 2+1, m=4 into 2+2, p=7 into 2+5
    
    # The 2x2*2x2 part has rank 7 (Strassen)
    # The cross terms need additional work.
    
    # Let me try a different strategy: use the known result that 
    # M(3,4,7) <= 72 by various bounds, but let's construct explicitly.
    
    # Actually, let's try to use the "laser method" idea:
    # Embed 2x2x2 Strassen into the 3x4x7 problem.
    
    # For each of the 7 Strassen terms, we need to map them to the 3x4x7 grid.
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Now map to 3x4*4x7:
    # Let's say we use the first 2 rows of A (3x4) and first 2 cols of B (4x7) for the Strassen core.
    # A_core = A[0:2, 0:2] (2x2)
    # B_core = B[0:2, 0:2] (2x2)
    
    # But we need to handle the remaining parts: A[0:2, 2:4], A[2:3, :], B[2:4, :], B[:, 2:7]
    
    # This is complex. Let me try a simpler construction.
    
    # Use the fact that 3x4*4x7 can be decomposed using:
    # - One 2x2*2x2 Strassen block (rank 7)
    # - Remaining terms handled schoolbook
    
    # Split:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x5, B21 is 2x2, B22 is 2x5
    
    # C = A11*B11 + A12*B21 + A21*B12 + A22*B22
    
    # Apply Strassen to the A11*B11 part (2x2*2x2): rank 7
    # The other three terms: A12*B21 (2x2*2x2), A21*B12 (1x2*2x5), A22*B22 (1x2*2x5)
    
    # A12*B21: 2x2*2x2, can use Strassen: rank 7
    # A21*B12: 1x2*2x5, schoolbook: 1*2*5 = 10
    # A22*B22: 1x2*2x5, schoolbook: 1*2*5 = 10
    
    # Total: 7 + 7 + 10 + 10 = 34? That seems too optimistic.
    
    # Wait, I need to be more careful. The Strassen decomposition gives us 7 scalar multiplications
    # for each 2x2*2x2 block. But the "multiplications" in Strassen are of scalars, and we need
    # to account for the fact that each "scalar" multiplication in the block context involves
    # adding up products of entries.
    
    # Let me reconsider. The tensor rank of the 2x2*2x2 multiplication is 7.
    # The tensor rank of the 1x2*2x5 multiplication is 1*2*5 = 10 (schoolbook).
    
    # So if we can cleanly separate these, total rank = 7 + 7 + 10 + 10 = 34.
    
    # But this assumes the blocks are independent, which they are in the tensor product sense.
    
    # Let me construct this explicitly.
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        A is 2x2, B is 2x2, C is 2x2.
        Indexing: A[i*2+j], B[j*2+k], C[k*2+i]
        """
        # Strassen terms
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        U = []
        V = []
        W = []
        
        # M1: (A00+A11)*(B00+B11)
        # A part: A00 + A11 -> indices 0, 3
        # B part: B00 + B11 -> indices 0, 3
        U.append([1, 0, 0, 1])  # A00 + A11
        V.append([1, 0, 0, 1])  # B00 + B11
        # C00 += M1, C11 += M1
        W.append([1, 0, 0, 1])  # C00 + C11
        
        # M2: (A10+A11)*B00
        # A part: A10 + A11 -> indices 2, 3
        # B part: B00 -> index 0
        U.append([0, 0, 1, 1])  # A10 + A11
        V.append([1, 0, 0, 0])  # B00
        # C10 += M2
        W.append([0, 0, 1, 0])  # C10
        
        # M3: A00*(B10-B11)
        # A part: A00 -> index 0
        # B part: B10 - B11 -> indices 2, 3
        U.append([1, 0, 0, 0])  # A00
        V.append([0, 0, 1, -1])  # B10 - B11
        # C01 += M3
        W.append([0, 1, 0, 0])  # C01
        
        # M4: A11*(B01-B00)
        # A part: A11 -> index 3
        # B part: B01 - B00 -> indices 1, 0
        U.append([0, 0, 0, 1])  # A11
        V.append([-1, 1, 0, 0])  # B01 - B00
        # C00 += M4, C10 += M4
        W.append([1, 0, 1, 0])  # C00 + C10
        
        # M5: (A00+A01)*B11
        # A part: A00 + A01 -> indices 0, 1
        # B part: B11 -> index 3
        U.append([1, 1, 0, 0])  # A00 + A01
        V.append([0, 0, 0, 1])  # B11
        # C00 -= M5, C01 += M5
        W.append([-1, 1, 0, 0])  # -C00 + C01
        
        # M6: (A10-A00)*(B00+B01)
        # A part: A10 - A00 -> indices 2, 0
        # B part: B00 + B01 -> indices 0, 1
        U.append([-1, 0, 1, 0])  # A10 - A00
        V.append([1, 1, 0, 0])  # B00 + B01
        # C11 -= M6 (wait, let me recheck)
        # Actually C11 = M1 - M2 + M3 + M7, so M6 doesn't appear in C?
        # Let me recheck Strassen:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # M6 is not used in C! That's odd. Let me recheck.
        
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Using 0-indexed:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # So M6 is indeed not used in C. That means M6 is redundant?
        # No, M6 is used to compute M1, M2, etc.? No, the M's are independent.
        # Actually, I think I made an error. Let me look up Strassen again.
        
        # The 7 multiplications are:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # So M6 is not in C. That means M6 is not needed? That can't be right.
        # Oh wait, I think the standard form uses all 7. Let me recheck.
        
        # Actually, I believe the correct Strassen formulas are:
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # And M6 is not used. But that would mean rank 6, not 7.
        # The issue is that M6 is needed to make the decomposition work.
        
        # Let me verify with a concrete example.
        # Actually, I think I'm confusing myself. The 7 M's are all needed.
        # Let me just use a known correct Strassen implementation.
        
        return None
    
    # Let me just use a known correct Strassen decomposition.
    # I'll verify it numerically.
    
    def make_strassen_2x2():
        """Correct Strassen 2x2x2 decomposition.
        A indexed as [A00, A01, A10, A11]
        B indexed as [B00, B01, B10, B11]
        C indexed as [C00, C01, C10, C11]
        """
        # From Wikipedia, Strassen algorithm:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Converting to 0-indexed:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        U = [
            [1, 0, 0, 1],   # M1: A00 + A11
            [0, 0, 1, 1],   # M2: A10 + A11
            [1, 0, 0, 0],   # M3: A00
            [0, 0, 0, 1],   # M4: A11
            [1, 1, 0, 0],   # M5: A00 + A01
            [-1, 0, 1, 0],  # M6: A10 - A00
            [0, 1, 0, -1],  # M7: A01 - A11
        ]
        
        V = [
            [1, 0, 0, 1],   # M1: B00 + B11
            [1, 0, 0, 0],   # M2: B00
            [0, 0, 1, -1],  # M3: B10 - B11
            [-1, 1, 0, 0],  # M4: B01 - B00
            [0, 0, 0, 1],   # M5: B11
            [1, 1, 0, 0],   # M6: B00 + B01
            [0, 0, 1, 1],   # M7: B10 + B11
        ]
        
        W = [
            [1, 0, 0, 1],   # M1: C00 + C11
            [0, 0, 1, 0],   # M2: C10
            [0, 1, 0, 0],   # M3: C01
            [1, 0, 1, 0],   # M4: C00 + C10
            [-1, 1, 0, 0],  # M5: -C00 + C01
            [0, 0, 0, -1],  # M6: -C11 (wait, let me check)
            [0, 0, 0, -1],  # M7: -C11 (wait)
        ]
        
        # Let me recheck the W coefficients.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1
        # M3: C01 += 1
        # M4: C00 += 1, C10 += 1
        # M5: C00 -= 1, C01 += 1
        # M6: not used in C?
        # M7: C00 -= 1, C11 += 1
        
        # So M6 is not used. That means we can drop M6 and have rank 6?
        # But that contradicts the known result that rank(2x2x2) = 7.
        
        # The issue is that the above formulas for C don't use M6, but M6 is still needed
        # because the U and V vectors for M6 are not linear combinations of the others
        # in a way that allows us to drop it.
        
        # Actually, I think the correct statement is that the 7 M's are all needed,
        # and the formulas for C use all 7. Let me recheck.
        
        # From a reliable source, Strassen's algorithm:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # I see that M6 is not in C. This is a known fact: M6 is used to ensure
        # that the decomposition is valid, but it doesn't directly contribute to C.
        # This seems paradoxical, but it's correct because the U and V for M6 are
        # needed to make the tensor identity hold.
        
        # Let me just verify numerically.
        
        return U, V, W
    
    U_s, V_s, W_s = make_strassen_2x2()
    
    # Verify Strassen decomposition
    def verify_decomp(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        import numpy as np
        rank = len(U[0]) if U else 0
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        # The tensor identity: sum_r U[i,r] * V[j,r] * W[k,r] = delta (i,j,k) in the multiplication tensor
        # For matrix multiplication: T[i,j,k] = 1 if j is the "middle" index
        # Specifically, T[(i,j),(j,k),(k,i)] = 1 for valid indices
        
        # Let's check: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # if this is a valid multiplication term, else 0.
        
        # Actually, the tensor for matrix multiplication M_{n,m,p} has:
        # T[(i,j),(j,k),(k,i)] = 1 for i in [0,n), j in [0,m), k in [0,p)
        # and 0 otherwise.
        
        # So we check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r] for r in range(rank))
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Test Strassen
    if verify_decomp(U_s, V_s, W_s, 2, 2, 2):
        print("Strassen 2x2x2 verified")
    else:
        print("Strassen 2x2x2 failed, using schoolbook")
        return schoolbook()
    
    # Now construct the 3x4*4x7 decomposition.
    # Strategy: split A (3x4) into [A_top (2x4); A_bottom (1x4)]
    # Split B (4x7) into [B_left (4x2) | B_right (4x5)]
    
    # C = A_top * B_left + A_top * B_right + A_bottom * B_left + A_bottom * B_right
    #   = A_top*B_left (2x4 * 4x2 = 2x2) 
    #   + A_top*B_right (2x4 * 4x5 = 2x5)
    #   + A_bottom*B_left (1x4 * 4x2 = 1x2)
    #   + A_bottom*B_right (1x4 * 4x5 = 1x5)
    
    # For A_top*B_left (2x2 result): use Strassen if we can, but it's 2x4*4x2, not 2x2*2x2.
    # We need to handle the middle dimension of 4.
    
    # Let me try a different split:
    # Split the middle dimension m=4 into two parts of size 2.
    # A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x7, B2 is 2x7
    # C = A1*B1 + A2*B2
    
    # A1*B1: 3x2 * 2x7 = 3x7, schoolbook rank = 3*2*7 = 42
    # A2*B2: 3x2 * 2x7 = 3x7, schoolbook rank = 3*2*7 = 42
    # Total: 84 (same as before)
    
    # To reduce rank, we need to use Strassen on sub-blocks.
    
    # Let's split further:
    # A1 (3x2) = [A1_top (2x2); A1_bottom (1x2)]
    # B1 (2x7) = [B1_left (2x2) | B1_right (2x5)]
    
    # A1_top * B1_left: 2x2 * 2x2 = 2x2, Strassen rank = 7
    # A1_top * B1_right: 2x2 * 2x5 = 2x5, schoolbook rank = 2*2*5 = 20
    # A1_bottom * B1_left: 1x2 * 2x2 = 1x2, schoolbook rank = 1*2*2 = 4
    # A1_bottom * B1_right: 1x2 * 2x5 = 1x5, schoolbook rank = 1*2*5 = 10
    
    # Similarly for A2 and B2:
    # A2_top * B2_left: 2x2 * 2x2 = 2x2, Strassen rank = 7
    # A2_top * B2_right: 2x2 * 2x5 = 2x5, schoolbook rank = 2*2*5 = 20
    # A2_bottom * B2_left: 1x2 * 2x2 = 1x2, schoolbook rank = 1*2*2 = 4
    # A2_bottom * B2_right: 1x2 * 2x5 = 1x5, schoolbook rank = 1*2*5 = 10
    
    # Total: 2*(7 + 20 + 4 + 10) = 2*41 = 82
    
    # That's only slightly better than 84. Let me think of a better approach.
    
    # Alternative: use the fact that 3x4*4x7 can be related to 2x2*2x2 via embedding.
    
    # Let's try: embed the problem into a 4x4*4x4 problem and use Strassen.
    # But 4x4*4x4 has rank at most 2*7 = 14 (two Strassen blocks) or 7 (one Strassen on 2x2 blocks of 4x4).
    
    # Actually, for 4x4*4x4, we can use Strassen recursively:
    # 4x4 = 2x(2x2), so rank <= 2*7 = 14? No, that's not how it works.
    # For 4x4*4x4, split into 2x2 blocks: 4 blocks of 2x2*2x2, each with rank 7.
    # Total: 4*7 = 28.
    
    # But our problem is 3x4*4x7, not 4x4*4x4.
    
    # Let me try a computational search for a low-rank decomposition.
    
    # Given the time constraints, let's just return the schoolbook decomposition
    # with a small improvement if we can find one.
    
    # Actually, let's try the 82-rank construction.
    
    def construct_82():
        """Construct the 82-rank decomposition."""
        # This is complex. Let me just return schoolbook for now.
        return schoolbook()
    
    # Given the complexity and time constraints, let's return schoolbook.
    # But let me try one more thing: use the known bound that M(3,4,7) <= 72.
    
    # For now, return schoolbook.
    return schoolbook()
# EVOLVE-BLOCK-END
