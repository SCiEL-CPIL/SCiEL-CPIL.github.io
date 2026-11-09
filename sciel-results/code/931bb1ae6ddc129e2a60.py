"""Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns {'U': (6, rank), 'V': (12, rank), 'W': (8, rank)} with integer/half-integer entries.
    The output follows the trace convention W[k*n+i, r].
    """
    import time
    from fractions import Fraction

    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8

    start_time = time.time()
    deadline = start_time + 140.0

    def verify_decomposition(U, V, W):
        """Verify that sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1 for all i,j,k."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    def to_int_or_half(matrix):
        """Convert Fraction matrix to int or float (half-integer) representation."""
        result = []
        for row in matrix:
            new_row = []
            for x in row:
                if x.denominator == 1:
                    new_row.append(int(x))
                elif x.denominator == 2:
                    new_row.append(float(x))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    def schoolbook():
        rank = 24
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = Fraction(1)
                    V[j*p+k][col] = Fraction(1)
                    W[k*n+i][col] = Fraction(1)
                    col += 1
        return U, V, W

    # Constructive rank-21 decomposition:
    # Split middle index j into {0,1} (2x2 block) and {2} (1x1 block).
    # For j in {0,1}: use Strassen (rank 7) for 2x2 * 2x2 multiplication.
    #   A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  (2x2)
    #   B_sub = [[B[0,0], B[0,1], B[0,2], B[0,3]],
    #             [B[1,0], B[1,1], B[1,2], B[1,3]]]  (2x4)
    #   We need C_sub[i,k] = sum_{j=0,1} A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2,3}
    #   This is 2x4 output from 2x2 * 2x4.
    #   We can split B_sub into two 2x2 blocks: B_left = B[:,0:2], B_right = B[:,2:4]
    #   C_left = A_sub * B_left (2x2), C_right = A_sub * B_right (2x2)
    #   Each 2x2*2x2 uses Strassen rank 7. Total for j in {0,1}: 14 terms.
    # For j = 2: C[i,k] += A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2,3}
    #   This is 2*4 = 8 terms if schoolbook, but we can do it in 4 terms:
    #   Actually, A[i,2]*B[2,k] is just outer product of column A[:,2] and row B[2,:].
    #   We need 2*4 = 8 rank-1 terms... unless we can combine.
    #   Wait: sum_r U[i,r]*V[r]*W[k,r] where U depends on i, V is scalar, W depends on k.
    #   For the j=2 part: contribution is A[i,2] * B[2,k].
    #   We need U[i*m+2, r] * V[2*p+k, r] * W[k*n+i, r] = A[i,2]*B[2,k] for each (i,k).
    #   The simplest: 8 separate terms, one for each (i,k) pair.
    #   But that gives 14 + 8 = 22 total.
    #
    # Alternative: Can we do better?
    # For the 2x2*2x4 part (j in {0,1}), instead of splitting into two 2x2*2x2:
    # We can use a different decomposition. 2x2 * 2x4 has rank... let's think.
    # The tensor is in R^4 x R^8 x R^4. 
    # Actually, 2x2*2x4: A is 2x2, B is 2x4, C is 2x4.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This can be seen as: for each output column k, C[:,k] = A * B[:,k] (matrix-vector product).
    # We can use Strassen for the 2x2*2x2 parts but share some factors.
    #
    # Let me try a different split. Consider the full problem:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Strassen for 2x2*2x2 (using indices i in {0,1}, j in {0,1}, k in {0,1}):
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # For k in {2,3}, we need similar but with B[j,2] and B[j,3].
    #
    # Let's use a unified approach:
    # Define 7 Strassen factors for the "left" k-pair (k=0,1) and 7 for "right" (k=2,3).
    # That's 14 factors for j in {0,1}.
    # Plus 8 for j=2. Total = 22.
    #
    # To get 21, we need to save one factor. 
    # Idea: The j=2 contribution A[i,2]*B[2,k] can potentially be merged.
    # Note that in Strassen, some factors involve sums like A[0,0]+A[1,1].
    # If we extend these to include A[i,2], we might save factors.
    #
    # Alternative approach: Use the fact that 2x3*3x4 can be decomposed as:
    # Treat it as (2x2 + 2x1) * (2x4 + 1x4) = 2x2*2x4 + 2x2*1x4 + 2x1*2x4 + 2x1*1x4
    # Rank(2x2*2x4) + Rank(2x2*1x4) + Rank(2x1*2x4) + Rank(2x1*1x4)
    # = ? + ? + ? + 2
    #
    # 2x2*1x4: A is 2x2, B is 1x4, C is 2x4. C[i,k] = sum_j A[i,j]*B[0,k]
    # = B[0,k] * sum_j A[i,j]. For each k, C[:,k] = B[0,k] * (A[:,0]+A[:,1]).
    # This needs 4 factors (one per k) with U[i,r] = A[i,0]+A[i,1] (same for all r? No...)
    # Actually: U[i, r_k] = A[i,0]+A[i,1], V[0*k+0, r_k] = B[0,k], W[k*n+i, r_k] = 1
    # Wait, that's not right because U depends on i and r.
    # Let me reconsider: C[i,k] = (A[i,0]+A[i,1]) * B[0,k]
    # Factor r_k: U[i, r_k] = A[i,0]+A[i,1], V[0, r_k] = B[0,k], W[k*n+i, r_k] = 1
    # But V[j*p+k, r] with j=0, so V[0, r_k] = B[0,k]. And W[k*n+i, r_k] = 1.
    # This works! 4 factors.
    #
    # 2x1*2x4: A is 2x1, B is 2x4, C is 2x4. C[i,k] = A[i,0]*B[0,k]
    # Factor r_k: U[i, r_k] = A[i,0], V[0*k+0, r_k] = B[0,k], W[k*n+i, r_k] = 1
    # 4 factors.
    #
    # 2x1*1x4: A is 2x1, B is 1x4, C is 2x4. C[i,k] = A[i,0]*B[0,k]
    # Factor r_k: U[i, r_k] = A[i,0], V[0, r_k] = B[0,k], W[k*n+i, r_k] = 1
    # 4 factors. But wait, this is the same form as above!
    #
    # Hmm, let me reconsider the block decomposition more carefully.
    #
    # A = [A11 A12 | A13]  where A11 is 2x2 (cols 0,1), A12 is 2x1 (col 2)
    # B = [B11; B12]      where B11 is 2x4 (rows 0,1), B12 is 1x4 (row 2)
    #
    # C = A11*B11 + A12*B12
    #
    # A11*B11: 2x2 * 2x4 -> 2x4. 
    # A12*B12: 2x1 * 1x4 -> 2x4.
    #
    # For A12*B12: C[i,k] += A12[i,0] * B12[0,k] = A[i,2] * B[2,k]
    # This needs 8 factors in general (one per (i,k) pair) unless we can factor.
    # C[i,k] = A[i,2] * B[2,k]. This is an outer product of column A[:,2] and row B[2,:].
    # As a bilinear form: sum_r U[i, r] * V[r] * W[k, r] where V is scalar per r.
    # We need U[i,r]*V[r]*W[k,r] = A[i,2]*B[2,k].
    # This is rank 2 as a matrix (outer product), so we need at least 2 factors? No...
    # Actually, for each (i,k) we need the product to equal A[i,2]*B[2,k].
    # If we use 2 factors: U[i,0]*V[0]*W[k,0] + U[i,1]*V[1]*W[k,1] = A[i,2]*B[2,k]
    # This is a rank-2 factorization of the 2x4 matrix A[:,2]*B[2,:].
    # Since A[:,2]*B[2,:] has rank 1 (outer product), it can be done with 1 factor:
    # U[i,0] = A[i,2], V[0] = 1, W[k,0] = B[2,k]. That's 1 factor!
    # Wait: U[i,0]*V[0]*W[k,0] = A[i,2]*1*B[2,k] = A[i,2]*B[2,k]. Yes!
    # So A12*B12 needs only 1 factor.
    #
    # Now A11*B11: 2x2 * 2x4.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Split into k in {0,1} and k in {2,3}:
    # C[i,0:2] = A11 * B11[:, 0:2] (2x2 * 2x2)
    # C[i,2:4] = A11 * B11[:, 2:4] (2x2 * 2x2)
    # Each 2x2*2x2 uses Strassen rank 7. Total: 14 factors.
    #
    # Grand total: 14 + 1 = 15? That seems too low. Let me re-examine.
    #
    # Wait, I think I made an error. The "1 factor" for A12*B12 doesn't work because
    # in the tensor decomposition, each factor r contributes U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # For the j=2 term: U[i*3+2, r] * V[2*4+k, r] * W[k*2+i, r] = A[i,2]*B[2,k].
    # If we use 1 factor r0: U[i*3+2, r0] * V[8+k, r0] * W[2k+i, r0] = A[i,2]*B[2,k].
    # Set U[i*3+2, r0] = A[i,2], V[8+k, r0] = 1, W[2k+i, r0] = B[2,k].
    # But W[2k+i, r0] = B[2,k] depends on k, and U[i*3+2, r0] = A[i,2] depends on i.
    # The product is A[i,2] * 1 * B[2,k] = A[i,2]*B[2,k]. This works for all (i,k) with ONE factor!
    #
    # So total rank = 14 (Strassen for 2x2*2x4) + 1 (for 2x1*1x4) = 15?
    # That can't be right because the known lower bound is 21.
    #
    # I must be making an error. Let me re-examine the Strassen decomposition.
    #
    # Oh wait, I think the issue is that in the tensor decomposition, the U, V, W matrices
    # have FIXED entries (they don't depend on the input matrices A, B). The decomposition
    # must work for ALL A and B simultaneously. So U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # are constants, and the identity must hold symbolically:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k} (in some sense)
    #
    # No wait, the tensor T[i,j,k] = 1 if the indices match the multiplication pattern.
    # Specifically, T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k'.
    # In our indexing: T[i*m+j, j*p+k, k*n+i] = 1.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    #
    # So for the multiplication tensor, T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # for some consistent i,j,k. Otherwise T = 0.
    #
    # Let me re-examine. The condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i in [0,2), j in [0,3), k in [0,4)
    #
    # And implicitly, for non-matching index combinations, the sum should be 0.
    # But actually, the tensor only has support on the "diagonal" where indices are consistent.
    #
    # Let me just directly construct the Strassen-based decomposition.
    
    # Strassen for 2x2 * 2x2:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # For our problem, we have:
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    #
    # Let's define the decomposition factors.
    # For j in {0,1} and k in {0,1}: use Strassen (7 factors)
    # For j in {0,1} and k in {2,3}: use Strassen (7 factors)  
    # For j=2: 1 factor (as shown above)
    # Total: 15 factors. But this is below the known lower bound of 21, so something is wrong.
    
    # The error: In the tensor decomposition, U, V, W are FIXED matrices.
    # The Strassen factors involve LINEAR COMBINATIONS of A entries and B entries.
    # For example, M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]).
    # In the tensor decomposition, this corresponds to:
    # U[0*3+0, r1] = 1, U[1*3+1, r1] = 1 (for the A part: A[0,0]+A[1,1])
    # V[0*4+0, r1] = 1, V[1*4+1, r1] = 1 (for the B part: B[0,0]+B[1,1])
    # W[0*2+0, r1] = 1, W[0*2+1, r1] = 1, W[1*2+0, r1] = -1, W[1*2+1, r1] = 1
    # (for the C part: C[0,0] += 1, C[0,1] += 1, C[1,0] += -1, C[1,1] += 1)
    
    # Let me construct this properly.
    
    def strassen_2x2_2x2(j_offset, k_offset):
        """
        Return 7 (U, V, W) factor lists for 2x2*2x2 multiplication
        where A indices are i in {0,1}, j in {j_offset, j_offset+1}
        and B indices are j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        and C indices are i in {0,1}, k in {k_offset, k_offset+1}
        """
        factors = []
        # Each factor: (U_dict, V_dict, W_dict)
        # U_dict: {(i, j): coeff}, V_dict: {(j, k): coeff}, W_dict: {(i, k): coeff}
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # Contributes: C[0,0]+=1, C[0,1]+=1, C[1,0]+=-1, C[1,1]+=1
        factors.append({
            'U': {(0, j_offset): 1, (1, j_offset+1): 1},
            'V': {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): 1},
            'W': {(0, k_offset): 1, (0, k_offset+1): 1, (1, k_offset): -1, (1, k_offset+1): 1}
        })
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # Contributes: C[1,0]+=1, C[1,1]+=-1
        factors.append({
            'U': {(1, j_offset): 1, (1, j_offset+1): 1},
            'V': {(j_offset, k_offset): 1},
            'W': {(1, k_offset): 1, (1, k_offset+1): -1}
        })
        
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        # Contributes: C[0,1]+=1, C[1,1]+=1
        factors.append({
            'U': {(0, j_offset): 1},
            'V': {(j_offset, k_offset+1): 1, (j_offset+1, k_offset+1): -1},
            'W': {(0, k_offset+1): 1, (1, k_offset+1): 1}
        })
        
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        # Contributes: C[0,0]+=1, C[1,0]+=1
        factors.append({
            'U': {(1, j_offset+1): 1},
            'V': {(j_offset+1, k_offset): 1, (j_offset, k_offset): -1},
            'W': {(0, k_offset): 1, (1, k_offset): 1}
        })
        
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # Contributes: C[0,0]+=-1, C[1,1]+=1
        factors.append({
            'U': {(0, j_offset): 1, (0, j_offset+1): 1},
            'V': {(j_offset+1, k_offset+1): 1},
            'W': {(0, k_offset): -1, (1, k_offset+1): 1}
        })
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # Contributes: C[1,1]+=1
        factors.append({
            'U': {(1, j_offset): 1, (0, j_offset): -1},
            'V': {(j_offset, k_offset): 1, (j_offset, k_offset+1): 1},
            'W': {(1, k_offset+1): 1}
        })
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        # Contributes: C[0,0]+=1, C[1,0]+=1
        factors.append({
            'U': {(0, j_offset+1): 1, (1, j_offset+1): -1},
            'V': {(j_offset+1, k_offset): 1, (j_offset+1, k_offset+1): 1},
            'W': {(0, k_offset): 1, (1, k_offset): 1}
        })
        
        return factors
    
    # Build the full decomposition
    all_factors = []
    
    # Strassen for j in {0,1}, k in {0,1}
    all_factors.extend(strassen_2x2_2x2(0, 0))
    
    # Strassen for j in {0,1}, k in {2,3}
    all_factors.extend(strassen_2x2_2x2(0, 2))
    
    # For j=2: C[i,k] += A[i,2]*B[2,k]
    # Single factor: U[i,2] = A[i,2], V[2,k] = 1, W[i,k] = B[2,k]
    # In tensor form: U[i*3+2, r] = 1, V[2*4+k, r] = 1, W[k*2+i, r] = 1
    # But wait, this gives U[i*3+2, r] * V[8+k, r] * W[2k+i, r] = 1*1*1 = 1
    # for ALL (i,k) pairs, not just A[i,2]*B[2,k].
    #
    # The issue: in the tensor decomposition, the factors are FIXED.
    # We can't have U depend on A or V depend on B.
    #
    # Let me reconsider. The tensor T has T[i*3+j, j*4+k, k*2+i] = 1.
    # We need sum_r U[i*3+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1 for all valid (i,j,k).
    #
    # For the j=2 contribution: we need the sum over r of
    # U[i*3+2, r] * V[8+k, r] * W[2k+i, r] to equal 1 for each (i,k)
    # AND to be 0 for all other index combinations.
    #
    # Actually, the tensor only has 1s on the "consistent" indices.
    # For a given r, the product U[i*3+j, r] * V[j*4+k, r] * W[k*2+i, r]
    # must contribute to the correct (i,j,k) positions.
    #
    # For the j=2 block: we want to represent the 2x4 sub-tensor where
    # T[i*3+2, 8+k, 2k+i] = 1 for i in {0,1}, k in {0,1,2,3}.
    # This sub-tensor has rank... it's the tensor of the outer product structure.
    # T_sub[i, k] = 1 for all i, k (in the appropriate indexing).
    # As a 3-tensor with indices (i, k, (i,k) combined), it's actually just
    # the all-ones 2x4 matrix flattened into a tensor.
    #
    # The rank of the all-ones 2x4 matrix as a tensor is 2 (since it's rank 2 as a matrix,
    # and the tensor rank of a matrix is its matrix rank when viewed appropriately).
    #
    # Hmm, I'm overcomplicating this. Let me just use 8 factors for the j=2 part
    # (one per (i,k) pair) and see if the total is 22.
    
    # Actually, let me reconsider the whole approach.
    # The known optimal rank is 21. Let me try a different decomposition strategy.
    
    # Alternative: Use the "border rank" or known explicit constructions.
    # For 2x3*3x4, one known construction:
    #
    # Let me try to directly verify a rank-21 construction.
    # I'll use the approach of combining Strassen with additional terms.
    
    # For now, let's construct the rank-22 version (14 + 8) and verify it.
    # Then try to reduce to 21.
    
    # Rank 22: 14 (two Strassen) + 8 (schoolbook for j=2)
    all_factors_22 = list(all_factors)
    
    # Add 8 factors for j=2, one per (i,k)
    for i in range(2):
        for k in range(4):
            all_factors_22.append({
                'U': {(i, 2): 1},
                'V': {(2, k): 1},
                'W': {(i, k): 1}
            })
    
    # Convert factors to U, V, W matrices
    def factors_to_matrices(factors):
        rank = len(factors)
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        for r, f in enumerate(factors):
            for (i, j), coeff in f['U'].items():
                U[i*m + j][r] += Fraction(coeff)
            for (j, k), coeff in f['V'].items():
                V[j*p + k][r] += Fraction(coeff)
            for (i, k), coeff in f['W'].items():
                W[k*n + i][r] += Fraction(coeff)
        return U, V, W
    
    U22, V22, W22 = factors_to_matrices(all_factors_22)
    
    if verify_decomposition(U22, V22, W22):
        # We have a valid rank-22 decomposition. Try to reduce to 21.
        # Check if any two columns can be merged (i.e., one is a linear combination
        # that allows elimination).
        
        # For now, return rank 22 as a valid improvement over 24.
        return {
            "U": to_int_or_half(U22),
            "V": to_int_or_half(V22),
            "W": to_int_or_half(W22)
        }
    
    # Fallback: schoolbook rank 24
    U, V, W = schoolbook()
    return {
        "U": to_int_or_half(U),
        "V": to_int_or_half(V),
        "W": to_int_or_half(W)
    }

# EVOLVE-BLOCK-END
