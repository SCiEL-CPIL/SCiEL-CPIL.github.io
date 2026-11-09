"""Search for a lower-rank tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import sys
import random
import math

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    def make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p."""
        rank = n * m * p  # 144
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is valid."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Start with schoolbook
    best_result = make_schoolbook()
    best_rank = n * m * p

    # Try to find a lower rank decomposition
    # Strategy: Use the structure of the problem.
    # The tensor is T_{i,j,k} = e_i^T A e_j * e_j^T B e_k
    # Actually, the matrix multiplication tensor M_{i,j,k} is defined such that
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T = sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} C^*_{k,i}
    # where M_{i,j,k} = delta_{j,j} (just 1 if the indices match the multiplication structure)
    # Wait, the tensor is just the structural tensor of matrix multiplication.
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # In the decomposition, we need:
    # sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = 1  for all i,j,k where the term exists.
    # But the term exists for ALL i, j, k?
    # No. The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # The tensor element T_{i,j,k} is 1 if A_{i,j} and B_{j,k} multiply to form part of C_{i,k}.
    # This is always true for any i,j,k in the dimensions.
    # So we need sum_r U_{i,j} V_{j,k} W_{k,i} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the indices of W are (k*n+i) which is k*n+i.
    # The product is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # And this sum over r must be 1 for ALL i, j, k.
    # This is a very specific constraint.
    # Let's check if rank 144 is indeed minimal or if we can do better.
    # For n=m=p=2, rank is 7 (Strassen) vs 8 (schoolbook).
    # For 4x4*4x4, rank is 49 (Lathrop, Pan) vs 64.
    # For 4x4*4x9, it's a rectangular case.
    # There are known results for rectangular matrix multiplication.
    # However, finding the exact minimal rank is an open problem for most sizes.
    # The seed program returns 144.
    # Can we find a decomposition with rank < 144?
    # If we can find ANY valid decomposition with rank < 144, the score will improve.
    # Since I don't have a known lower-rank decomposition for 4x4*4x9,
    # I will try a heuristic search for a decomposition with rank < 144.
    # But searching for a rank-r decomposition is an NP-hard problem (in general).
    # Given the time limit, a random search is unlikely to succeed.
    # However, we can try to exploit symmetries or block structures.
    
    # Let's try a block Strassen-like approach.
    # Split A into 2x2 blocks of 2x2 matrices.
    # Split B into 2x2 blocks of 2x9 matrices.
    # This doesn't map directly to standard Strassen.
    
    # Alternative: Since 9 is not a power of 2, maybe we can split 9 into 4+4+1.
    # A (4x4) * B (4x9) = A * [B1 (4x4) | B2 (4x4) | B3 (4x1)]
    # C = [A*B1 (4x4) | A*B2 (4x4) | A*B3 (4x1)]
    # Rank(A*B1) + Rank(A*B2) + Rank(A*B3)
    # If we use Strassen for 4x4*4x4, rank is 49 each.
    # For 4x4*4x1, this is matrix-vector multiplication.
    # Rank of 4x4*4x1 is 4*4*1 = 16? Or can we do better?
    # Actually, 4x4*4x1 is just a linear map from R^4 to R^4, applied to a vector.
    # The tensor rank of 4x4*4x1 is the rank of the 4x4 matrix? No.
    # The tensor is T_{i,j,k} with k=0 only.
    # sum_r U_{i,j} V_{j,0} W_{0,i} = 1 for all i,j.
    # This is equivalent to finding a rank-r decomposition of the identity matrix?
    # No, it's sum_j U_{i,j} V_{j,0} W_{0,i} = 1.
    # Let x_j = V_{j,0}, y_i = W_{0,i}.
    # Then sum_j U_{i,j} x_j y_i = 1.
    # This implies y_i * (U x)_i = 1.
    # So (U x)_i = 1/y_i.
    # We can choose y_i = 1, then U x = 1 (vector of ones).
    # We need to decompose this into rank r.
    # U is 4x4, x is 4x1.
    # This is just a linear system. The "rank" here is the number of terms.
    # If we set r=4, we can easily do this.
    # So the 4x4*4x1 part has rank at most 16 (schoolbook) but likely less.
    # Actually, the schoolbook rank for 4x4*4x1 is 4*4*1 = 16.
    # Can we do it in rank 4?
    # Let r=4. U_{i,j}, V_{j,0}, W_{0,i}.
    # sum_{j=1..4} U_{i,j} V_{j,0} W_{0,i} = 1.
    # Let W_{0,i} = 1. Then sum_j U_{i,j} V_{j,0} = 1.
    # Let V_{j,0} = 1. Then sum_j U_{i,j} = 1.
    # We can choose U to be any matrix with row sums 1.
    # So rank 4 is sufficient for 4x4*4x1.
    # Wait, the indices are:
    # U has shape (n*m, rank) = (16, r)
    # V has shape (m*p, rank) = (36, r)
    # W has shape (p*n, rank) = (36, r)
    # For the 4x4*4x1 part (k=0):
    # U[i*4+j, r] * V[j*9+0, r] * W[0*4+i, r]
    # Sum over r = 1.
    # If we dedicate 4 terms to this, we can set:
    # For r=0..3:
    #   U[i*4+r, r] = 1 (for all i)
    #   V[r*9+0, r] = 1 (for all j=r? No, V index is j*9+0)
    #   Let's be careful.
    #   We need sum_r U_{i,r} V_{r,0} W_{0,i} = 1. (Using r as index 0..3)
    #   Let W_{0,i} = 1 for all i.
    #   Let V_{j,0} = 1 for all j? No, V index is j*p+k. Here k=0, so V[j*9+0, r].
    #   We need sum_r U[i*4+j, r] V[j*9, r] W[i, r] = 1.
    #   If we set r=0..3:
    #     U[i*4+0, 0] = 1, others 0?
    #     Then sum_r ... = U[i*4+0, 0] V[0*9, 0] W[i, 0] + ...
    #     This doesn't work for all j.
    #   We need the sum to be 1 for ALL i, j.
    #   If we use rank 4, we can't cover all j independently with only 4 terms unless we use a specific structure.
    #   Actually, the "rank" of the 4x4*4x1 tensor is 4.
    #   Proof: The tensor is T_{i,j} = 1 (for k=0).
    #   This is the matrix of all ones, which has rank 1.
    #   So we can do it in rank 1?
    #   sum_r U_{i,j} V_{j,0} W_{0,i} = 1.
    #   Let r=0.
    #   U_{i,j} = 1, V_{j,0} = 1, W_{0,i} = 1.
    #   Sum = 1*1*1 = 1.
    #   Yes! Rank 1 is sufficient for 4x4*4x1.
    #   So the 4x4*4x1 part costs 1 rank.
    
    # So, if we split 9 into 4+4+1:
    # C = A*B1 (4x4) + A*B2 (4x4) + A*B3 (4x1)
    # Rank(A*B1) = 49 (Strassen 4x4)
    # Rank(A*B2) = 49
    # Rank(A*B3) = 1
    # Total rank = 49 + 49 + 1 = 99.
    # This is much better than 144!
    # We need to construct this explicitly.
    
    # Strassen for 4x4*4x4:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each M is a 2x2*2x2 product.
    # Rank of 2x2*2x2 is 7 (Strassen) or 4 (schoolbook).
    # If we use Strassen for the 2x2 blocks, each M costs 7 ranks.
    # Total 7 M's * 7 ranks = 49 ranks.
    
    # So we need to implement Strassen 2x2*2x2.
    # The 2x2*2x2 tensor has rank 7.
    # We can find a known rank-7 decomposition for 2x2*2x2.
    # One standard decomposition (Strassen):
    # Terms 0-6.
    # Let's find the U, V, W matrices for 2x2*2x2 rank 7.
    # Indices: i,j,k in {0,1}.
    # U is 4x7, V is 4x7, W is 4x7.
    # (Using 0-based indexing for 2x2 matrices)
    
    # Strassen 2x2 decomposition:
    # M1 = (a11+a22)(b11+b22) -> U: [1,0,0,1] (for a11,a22), V: [1,0,0,1] (for b11,b22), W: [1,0,0,1] (for c11,c22) - wait, W is for C.
    # Actually, the tensor is T_{i,j,k}.
    # For Strassen, the 7 terms are:
    # 1: (A11+A22)(B11+B22) -> contributes to C11, C22
    # 2: (A21+A22)B11 -> contributes to C21
    # 3: A11(B12-B22) -> contributes to C12
    # 4: A22(B21-B11) -> contributes to C21
    # 5: (A11+A12)B22 -> contributes to C12
    # 6: (A21-A11)(B11+B12) -> contributes to C22
    # 7: (A12-A22)(B21+B22) -> contributes to C22
    
    # Let's write the U, V, W for each term (1-indexed for Strassen, 0-indexed for Python).
    # A indices: 00, 01, 10, 11 -> flat: 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> flat: 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> flat: 0, 1, 2, 3
    
    # Term 0 (Strassen 1): (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (contributes to C00 and C11)
    # Wait, W is [1,0,0,1] means it contributes to C00 and C11.
    # But the term is (A00+A11)(B00+B11).
    # This product is added to C00 and C11?
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes to C00 and C11.
    # So W[0] = 1, W[3] = 1.
    
    # Term 1 (Strassen 2): (A10+A11)B00
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 0, 1, 0] (contributes to C10)
    
    # Term 2 (Strassen 3): A00(B01-B11)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: [0, 1, 0, 0] (contributes to C01)
    
    # Term 3 (Strassen 4): A11(B10-B00)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, -1]
    # W: [0, 0, 1, 0] (contributes to C10)
    
    # Term 4 (Strassen 5): (A00+A01)B11
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: [0, 1, 0, 0] (contributes to C01)
    
    # Term 5 (Strassen 6): (A10-A00)(B00+B01)
    # U: [0, 0, 1, -1]  # Wait, A10-A00 is [0,0,1,-1]
    # V: [1, 1, 0, 0]
    # W: [0, 0, 0, 1] (contributes to C11)
    
    # Term 6 (Strassen 7): (A01-A11)(B10+B11)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: [0, 0, 0, 1] (contributes to C11)
    
    # Let's verify this decomposition.
    # C00 = T0*W0[0] + T3*W3[0] + ...
    # T0 = (A00+A11)(B00+B11)
    # T3 = A11(B10-B00)
    # T4 = (A00+A01)B11
    # T6 = (A01-A11)(B10+B11)
    # C00 should be A00*B00 + A01*B10.
    # Sum = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + (A00B11 - A00B11) + (A11B00 - A11B00) + (A11B11 - A11B11) + (A11B10 - A11B10) + (-A01B11 + A01B11) + A01B10
    # = A00B00 + A01B10. Correct!
    
    # C11 = T0*W0[3] - T1*W1[3] + T2*W2[3] + T5*W5[3]
    # T0 = (A00+A11)(B00+B11)
    # T1 = (A10+A11)B00
    # T2 = A00(B01-B11)
    # T5 = (A10-A00)(B00+B01)
    # C11 should be A10*B01 + A11*B11.
    # Sum = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = (A00B00 - A00B00) + (A00B11 - A00B11) + (A11B00 - A11B00) + A11B11 + (-A10B00 + A10B00) + (A00B01 - A00B01) + A10B01
    # = A11B11 + A10B01. Correct!
    
    # So the 2x2*2x2 rank 7 decomposition is valid.
    
    # Now we need to assemble the 4x4*4x9 decomposition.
    # Split A into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # Split B into 2x4 blocks: B0 (4x4), B1 (4x4), B2 (4x1)
    # B0 = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2)
    # B1 = [[B20, B21], [B30, B31]] (2x2 blocks of 2x2)
    # B2 = [[B40], [B50]] (2x1 blocks of 2x1)
    
    # C = A*B = [C0 | C1 | C2]
    # C0 = A*B0 (4x4)
    # C1 = A*B1 (4x4)
    # C2 = A*B2 (4x1)
    
    # C0 uses Strassen 4x4*4x4 (rank 49)
    # C1 uses Strassen 4x4*4x4 (rank 49)
    # C2 uses rank 1 decomposition (rank 1)
    # Total rank = 99.
    
    # We need to map the indices.
    # A is 4x4. Flat index: i*4+j.
    # A00: i in {0,1}, j in {0,1} -> flat 0,1,4,5
    # A01: i in {0,1}, j in {2,3} -> flat 2,3,6,7
    # A10: i in {2,3}, j in {0,1} -> flat 8,9,12,13
    # A11: i in {2,3}, j in {2,3} -> flat 10,11,14,15
    
    # B is 4x9. Flat index: j*9+k.
    # B0: k in {0..3}
    #   B00: j in {0,1}, k in {0,1} -> flat 0,1, 9,10
    #   B01: j in {0,1}, k in {2,3} -> flat 2,3, 11,12
    #   B10: j in {2,3}, k in {0,1} -> flat 18,19, 27,28
    #   B11: j in {2,3}, k in {2,3} -> flat 20,21, 29,30
    # B1: k in {4..7}
    #   B20: j in {0,1}, k in {4,5} -> flat 4,5, 13,14
    #   B21: j in {0,1}, k in {6,7} -> flat 6,7, 15,16
    #   B30: j in {2,3}, k in {4,5} -> flat 22,23, 31,32
    #   B31: j in {2,3}, k in {6,7} -> flat 24,25, 33,34
    # B2: k in {8}
    #   B40: j in {0,1}, k in {8} -> flat 8, 17
    #   B50: j in {2,3}, k in {8} -> flat 26, 35
    
    # C is 4x9. Flat index: k*4+i. (Trace convention W[k*n+i,r])
    # C0: k in {0..3}
    #   C00: k in {0,1}, i in {0,1} -> flat 0,1, 4,5
    #   C01: k in {0,1}, i in {2,3} -> flat 2,3, 6,7
    #   C10: k in {2,3}, i in {0,1} -> flat 8,9, 12,13
    #   C11: k in {2,3}, i in {2,3} -> flat 10,11, 14,15
    # C1: k in {4..7}
    #   C20: k in {4,5}, i in {0,1} -> flat 16,17, 20,21
    #   C21: k in {4,5}, i in {2,3} -> flat 18,19, 22,23
    #   C30: k in {6,7}, i in {0,1} -> flat 24,25, 28,29
    #   C31: k in {6,7}, i in {2,3} -> flat 26,27, 30,31
    # C2: k in {8}
    #   C40: k in {8}, i in {0,1} -> flat 32, 33
    #   C50: k in {8}, i in {2,3} -> flat 34, 35
    
    # We will construct the U, V, W matrices for the 99-rank decomposition.
    # Rank 0-48: C0 (Strassen 4x4*4x4)
    # Rank 49-97: C1 (Strassen 4x4*4x4)
    # Rank 98: C2 (Rank 1)
    
    rank = 99
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Strassen 2x2 terms (U, V, W vectors of length 4)
    # U, V, W for the 7 terms of 2x2*2x2
    # Indices: 00, 01, 10, 11
    strassen_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, 0],  # M2 -> C10
        [0, 1, 0, 0],  # M3 -> C01
        [0, 0, 1, 0],  # M4 -> C10
        [0, 1, 0, 0],  # M5 -> C01
        [0, 0, 0, 1],  # M6 -> C11
        [0, 0, 0, 1]   # M7 -> C11
    ]
    
    # Mapping for 4x4*4x4 Strassen
    # A blocks: A00, A01, A10, A11
    # A00 flat indices: [0, 1, 4, 5]
    # A01 flat indices: [2, 3, 6, 7]
    # A10 flat indices: [8, 9, 12, 13]
    # A11 flat indices: [10, 11, 14, 15]
    
    # B blocks for B0 (k=0..3):
    # B00 flat indices: [0, 1, 9, 10]
    # B01 flat indices: [2, 3, 11, 12]
    # B10 flat indices: [18, 19, 27, 28]
    # B11 flat indices: [20, 21, 29, 30]
    
    # C blocks for C0 (k=0..3):
    # C00 flat indices: [0, 1, 4, 5]
    # C01 flat indices: [2, 3, 6, 7]
    # C10 flat indices: [8, 9, 12, 13]
    # C11 flat indices: [10, 11, 14, 15]
    
    # B blocks for B1 (k=4..7):
    # B20 flat indices: [4, 5, 13, 14]
    # B21 flat indices: [6, 7, 15, 16]
    # B30 flat indices: [22, 23, 31, 32]
    # B31 flat indices: [24, 25, 33, 34]
    
    # C blocks for C1 (k=4..7):
    # C20 flat indices: [16, 17, 20, 21]
    # C21 flat indices: [18, 19, 22, 23]
    # C30 flat indices: [24, 25, 28, 29]
    # C31 flat indices: [26, 27, 30, 31]
    
    def add_strassen_block(rank_offset, A_indices, B_indices, C_indices):
        """Add a Strassen 4x4*4x4 decomposition starting at rank_offset."""
        # A_indices: list of 4 lists of 4 flat indices for A00, A01, A10, A11
        # B_indices: list of 4 lists of 4 flat indices for B00, B01, B10, B11
        # C_indices: list of 4 lists of 4 flat indices for C00, C01, C10, C11
        
        for t in range(7):
            # M1
            u_vec = strassen_U[0]
            v_vec = strassen_V[0]
            w_vec = strassen_W[0]
            
            # Map u_vec to U
            for local_idx, val in enumerate(u_vec):
                if val != 0:
                    # local_idx 0->A00, 1->A01, 2->A10, 3->A11
                    # A00 has 4 indices: A_indices[0]
                    # But u_vec is length 4, corresponding to the 4 elements of the 2x2 matrix.
                    # Wait, the Strassen decomposition is for the 2x2 matrix elements.
                    # U[i,j] for i,j in {0,1}.
                    # So u_vec[0] is for A00[0,0], u_vec[1] for A00[0,1]? No.
                    # The Strassen 2x2 decomposition uses the 4 elements of the 2x2 matrix.
                    # U is a vector of length 4.
                    # U[0] corresponds to element (0,0) of the 2x2 block.
                    # U[1] corresponds to element (0,1).
                    # U[2] corresponds to element (1,0).
                    # U[3] corresponds to element (1,1).
                    
                    # We need to map these to the flat indices of the 4x4 matrix.
                    # A00 is a 2x2 block. Its flat indices are A_indices[0].
                    # A00[0,0] -> A_indices[0][0]
                    # A00[0,1] -> A_indices[0][1]
                    # A00[1,0] -> A_indices[0][2]
                    # A00[1,1] -> A_indices[0][3]
                    
                    # Which block does local_idx belong to?
                    # local_idx 0,1 -> A00? No.
                    # The Strassen 2x2 decomposition is applied to the 2x2 blocks of A.
                    # The 7 terms M1..M7 are products of 2x2 matrices.
                    # Each term involves a linear combination of the 2x2 blocks.
                    # E.g., M1 = (A00+A11)(B00+B11).
                    # This involves A00 and A11.
                    # So for M1, U has non-zero entries for A00 and A11.
                    # Specifically, U_A00 = 1, U_A11 = 1.
                    # But U_A00 is a 2x2 matrix of ones?
                    # No, the Strassen 2x2 decomposition is for the *elements* of the 2x2 matrix.
                    # Wait, Strassen is for 2x2 matrices.
                    # The tensor is for 2x2*2x2.
                    # The U, V, W vectors are of length 4 (the 4 elements).
                    # So for M1, U = [1, 0, 0, 1].
                    # This means U[0,0] = 1, U[1,1] = 1.
                    # So for the 4x4*4x4 case, M1 involves A00[0,0], A00[1,1], A11[0,0], A11[1,1].
                    # So we add to U[A00_idx[0]], U[A00_idx[3]], U[A11_idx[0]], U[A11_idx[3]].
                    
                    # We need to know which block and which local index.
                    # The strassen_U vectors are for the 2x2 block elements.
                    # But the 4x4*4x4 Strassen uses 2x2 *blocks*.
                    # So we need to apply the 2x2 decomposition to each pair of blocks?
                    # No, the 4x4*4x4 Strassen is a direct construction.
                    # It uses 7 multiplications of 2x2 matrices.
                    # Each 2x2 matrix multiplication has rank 7.
                    # So total rank 7*7=49.
                    
                    # We need to decompose each 2x2 matrix multiplication.
                    # There are 7 such multiplications.
                    # For each of the 7, we use the 2x2 rank-7 decomposition.
                    
                    # So, for M1 = (A00+A11)(B00+B11):
                    # This is a 2x2*2x2 multiplication.
                    # We decompose it into 7 terms.
                    # For each of these 7 terms, we have U, V, W vectors of length 4.
                    # These vectors correspond to the elements of the 2x2 matrices (A00+A11), (B00+B11), and the result.
                    
                    # So, for M1, term t (0..6):
                    # U_vec = strassen_U[t] (length 4)
                    # V_vec = strassen_V[t] (length 4)
                    # W_vec = strassen_W[t] (length 4)
                    
                    # U_vec[0] is for (A00+A11)[0,0] = A00[0,0] + A11[0,0]
                    # So we add U_vec[0] to U[A00_flat[0]] and U[A11_flat[0]].
                    
                    # Let's define the flat indices for the 2x2 blocks.
                    # A00_flat = [0, 1, 4, 5] -> (0,0), (0,1), (1,0), (1,1)
                    # A01_flat = [2, 3, 6, 7]
                    # A10_flat = [8, 9, 12, 13]
                    # A11_flat = [10, 11, 14, 15]
                    
                    # For M1, the A matrix is A00+A11.
                    # So the flat indices are A00_flat and A11_flat.
                    # U_vec[0] adds to A00_flat[0] and A11_flat[0].
                    # U_vec[1] adds to A00_flat[1] and A11_flat[1].
                    # etc.
                    
                    pass # Placeholder for complex logic
                    
    # The logic above is getting complicated.
    # Let's simplify by directly constructing the 49-rank decomposition for 4x4*4x4.
    # We know the 7 M's.
    # For each M, we use the 2x2 rank-7 decomposition.
    
    # Let's define the flat indices for the 2x2 blocks of A, B, C for C0.
    A00_f = [0, 1, 4, 5]
    A01_f = [2, 3, 6, 7]
    A10_f = [8, 9, 12, 13]
    A11_f = [10, 11, 14, 15]
    
    B00_f = [0, 1, 9, 10]
    B01_f = [2, 3, 11, 12]
    B10_f = [18, 19, 27, 28]
    B11_f = [20, 21, 29, 30]
    
    C00_f = [0, 1, 4, 5]
    C01_f = [2, 3, 6, 7]
    C10_f = [8, 9, 12, 13]
    C11_f = [10, 11, 14, 15]
    
    # The 7 M's and their A, B, C block combinations.
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B01-B11) -> C01
    # M4 = A11(B10-B00) -> C10
    # M5 = (A00+A01)B11 -> C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C11
    
    # For each M, we have a list of (A_blocks, B_blocks, C_blocks) with coefficients.
    # E.g., M1: A: [(A00, 1), (A11, 1)], B: [(B00, 1), (B11, 1)], C: [(C00, 1), (C11, 1)]
    
    M1_A = [(A00_f, 1), (A11_f, 1)]
    M1_B = [(B00_f, 1), (B11_f, 1)]
    M1_C = [(C00_f, 1), (C11_f, 1)]
    
    M2_A = [(A10_f, 1), (A11_f, 1)]
    M2_B = [(B00_f, 1)]
    M2_C = [(C10_f, 1)]
    
    M3_A = [(A00_f, 1)]
    M3_B = [(B01_f, 1), (B11_f, -1)]
    M3_C = [(C01_f, 1)]
    
    M4_A = [(A11_f, 1)]
    M4_B = [(B10_f, 1), (B00_f, -1)]
    M4_C = [(C10_f, 1)]
    
    M5_A = [(A00_f, 1), (A01_f, 1)]
    M5_B = [(B11_f, 1)]
    M5_C = [(C01_f, 1)]
    
    M6_A = [(A10_f, 1), (A00_f, -1)]
    M6_B = [(B00_f, 1), (B01_f, 1)]
    M6_C = [(C11_f, 1)]
    
    M7_A = [(A01_f, 1), (A11_f, -1)]
    M7_B = [(B10_f, 1), (B11_f, 1)]
    M7_C = [(C11_f, 1)]
    
    # Now, for each M, we decompose it using the 2x2 rank-7 decomposition.
    # The 2x2 decomposition has 7 terms.
    # For each term, we have U_vec, V_vec, W_vec of length 4.
    # These vectors correspond to the elements of the 2x2 matrices.
    # We need to add the contributions to the global U, V, W.
    
    def add_2x2_decomposition(rank_start, A_comb, B_comb, C_comb):
        """Add a 2x2*2x2 decomposition (rank 7) starting at rank_start."""
        for t in range(7):
            r = rank_start + t
            u_vec = strassen_U[t]
            v_vec = strassen_V[t]
            w_vec = strassen_W[t]
            
            # Add U contributions
            for local_idx, val in enumerate(u_vec):
                if val != 0:
                    for block_f, coeff in A_comb:
                        # local_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                        # block_f is a list of 4 flat indices in the same order.
                        U[block_f[local_idx]][r] += val * coeff
            
            # Add V contributions
            for local_idx, val in enumerate(v_vec):
                if val != 0:
                    for block_f, coeff in B_comb:
                        V[block_f[local_idx]][r] += val * coeff
            
            # Add W contributions
            for local_idx, val in enumerate(w_vec):
                if val != 0:
                    for block_f, coeff in C_comb:
                        W[block_f[local_idx]][r] += val * coeff
    
    # Add M1
    add_2x2_decomposition(0, M1_A, M1_B, M1_C)
    # Add M2
    add_2x2_decomposition(7, M2_A, M2_B, M2_C)
    # Add M3
    add_2x2_decomposition(14, M3_A, M3_B, M3_C)
    # Add M4
    add_2x2_decomposition(21, M4_A, M4_B, M4_C)
    # Add M5
    add_2x2_decomposition(28, M5_A, M5_B, M5_C)
    # Add M6
    add_2x2_decomposition(35, M6_A, M6_B, M6_C)
    # Add M7
    add_2x2_decomposition(42, M7_A, M7_B, M7_C)
    
    # Now for C1 (A*B1)
    # B1 blocks:
    B20_f = [4, 5, 13, 14]
    B21_f = [6, 7, 15, 16]
    B30_f = [22, 23, 31, 32]
    B31_f = [24, 25, 33, 34]
    
    C20_f = [16, 17, 20, 21]
    C21_f = [18, 19, 22, 23]
    C30_f = [24, 25, 28, 29]
    C31_f = [26, 27, 30, 31]
    
    # The A blocks are the same.
    # The structure is the same, just different B and C blocks.
    N1_A = [(A00_f, 1), (A11_f, 1)]
    N1_B = [(B20_f, 1), (B31_f, 1)]  # B20 is B00 equivalent, B31 is B11 equivalent
    N1_C = [(C20_f, 1), (C31_f, 1)]
    
    N2_A = [(A10_f, 1), (A11_f, 1)]
    N2_B = [(B20_f, 1)]
    N2_C = [(C30_f, 1)]
    
    N3_A = [(A00_f, 1)]
    N3_B = [(B21_f, 1), (B31_f, -1)]
    N3_C = [(C21_f, 1)]
    
    N4_A = [(A11_f, 1)]
    N4_B = [(B30_f, 1), (B20_f, -1)]
    N4_C = [(C30_f, 1)]
    
    N5_A = [(A00_f, 1), (A01_f, 1)]
    N5_B = [(B31_f, 1)]
    N5_C = [(C21_f, 1)]
    
    N6_A = [(A10_f, 1), (A00_f, -1)]
    N6_B = [(B20_f, 1), (B21_f, 1)]
    N6_C = [(C31_f, 1)]
    
    N7_A = [(A01_f, 1), (A11_f, -1)]
    N7_B = [(B30_f, 1), (B31_f, 1)]
    N7_C = [(C31_f, 1)]
    
    # Add N1..N7 starting at rank 49
    add_2x2_decomposition(49, N1_A, N1_B, N1_C)
    add_2x2_decomposition(56, N2_A, N2_B, N2_C)
    add_2x2_decomposition(63, N3_A, N3_B, N3_C)
    add_2x2_decomposition(70, N4_A, N4_B, N4_C)
    add_2x2_decomposition(77, N5_A, N5_B, N5_C)
    add_2x2_decomposition(84, N6_A, N6_B, N6_C)
    add_2x2_decomposition(91, N7_A, N7_B, N7_C)
    
    # Now for C2 (A*B2)
    # B2 is 4x1.
    # B40_f = [8, 17] (j=0,1; k=8)
    # B50_f = [26, 35] (j=2,3; k=8)
    
    # C2 is 4x1.
    # C40_f = [32, 33] (i=0,1; k=8)
    # C50_f = [34, 35] (i=2,3; k=8)
    
    # We need sum_r U[i,j,r] V[j,8,r] W[8,i,r] = 1 for all i,j.
    # We can do this in rank 1.
    # Let r=98.
    # U[i,j,98] = 1 for all i,j.
    # V[j,8,98] = 1 for all j.
    # W[8,i,98] = 1 for all i.
    
    for i in range(4):
        for j in range(4):
            U[i*4+j][98] = 1
    for j in range(4):
        V[j*9+8][98] = 1
    for i in range(4):
        W[8*4+i][98] = 1
    
    # Verify the result
    if verify(U, V, W):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    else:
        # Fallback to schoolbook if verification fails
        pass

    return best_result
# EVOLVE-BLOCK-END
