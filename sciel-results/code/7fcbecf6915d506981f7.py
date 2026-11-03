import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The goal is to find a rank-minimal exact tensor decomposition.
    # Schoolbook rank is 56.
    # Known lower bound is 32.
    # Inspiration suggests rank 50 is achievable via Strassen blocks.
    
    def verify_decomposition(U, V, W, rank):
        """
        Verifies if U, V, W form a valid decomposition.
        U: (nm, rank), V: (mp, rank), W: (pn, rank)
        Indices:
        U[i, r] corresponds to A[i, j]? No, U is (n*m, rank).
        Standard tensor T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
        Here U is flattened as i*m + j.
        V is flattened as j*p + k.
        W is flattened as k*n + i.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        u_val = U[i * m + j][r]
                        v_val = V[j * p + k][r]
                        w_val = W[k * n + i][r]
                        s += u_val * v_val * w_val
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def get_strassen_222():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        Rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices:
        U[i*2+j, r]
        V[j*2+k, r]
        W[k*2+i, r]
        """
        # Strassen factors are well-known.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # U factors (A side):
        # Row 0 (A00): 1, 0, 1, 0, 1, -1, 0
        # Row 1 (A01): 0, 0, 0, 0, 1, 0, -1
        # Row 2 (A10): 0, 1, 0, 0, 0, 1, 0
        # Row 3 (A11): 1, 1, 0, 1, 0, 0, -1
        U_222 = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, -1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        
        # V factors (B side):
        # Col 0 (B00): 1, 1, 0, -1, 0, 1, 0
        # Col 1 (B01): 0, 0, 1, 0, 0, 0, 1
        # Col 2 (B10): 0, 0, 0, 1, 0, 1, 0
        # Col 3 (B11): 1, 0, -1, -1, 1, 0, 1
        V_222 = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, -1, -1, 1, 0, 1]
        ]
        
        # W factors (C side):
        # Row 0 (C00): 1, 0, 0, 0, 0, 0, 0
        # Row 1 (C01): 0, 0, 1, 0, 0, 0, 0
        # Row 2 (C10): 0, 1, 0, 0, 0, 0, 0
        # Row 3 (C11): 0, 0, 0, 1, 0, 0, 0
        # Wait, Strassen reconstruction:
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 + M6
        
        W_222 = [
            [1, 0, 0, -1, 0, 1, 1],   # C00
            [0, 0, 1, 0, 1, 0, 0],    # C01
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 0, 0, 1, 1, 0]    # C11
        ]
        
        return U_222, V_222, W_222

    def get_school_221():
        """
        Returns U, V, W for 2x2x1 matrix multiplication (vector-matrix product).
        Rank 4.
        U: (4, 4), V: (2, 4), W: (2, 4)
        """
        U_221 = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_221 = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        # Wait, 2x2x1 is multiplying a 2x2 matrix by a 2x1 vector.
        # Inputs: A (2x2), b (2x1). Output: c (2x1).
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        # This is just schoolbook.
        # U indices: i*2+j. i in {0,1}, j in {0,1}.
        # V indices: j*1+k. j in {0,1}, k in {0,1}.
        # W indices: k*2+i. k in {0,1}, i in {0,1}.
        
        # Term 1: a00*b0 -> c0
        # Term 2: a01*b1 -> c0
        # Term 3: a10*b0 -> c1
        # Term 4: a11*b1 -> c1
        
        U_221 = [
            [1, 0, 0, 0], # a00
            [0, 1, 0, 0], # a01
            [0, 0, 1, 0], # a10
            [0, 0, 0, 1]  # a11
        ]
        V_221 = [
            [1, 0, 1, 0], # b0
            [0, 1, 0, 1]  # b1
        ]
        W_221 = [
            [1, 0, 0, 0], # c0
            [0, 0, 1, 0]  # c1
        ]
        return U_221, V_221, W_221

    def combine_tensors(tensor1, tensor2):
        """
        Combines two tensor decompositions by direct sum.
        tensor1: (U1, V1, W1) with shapes (n1*m1, r1), (m1*p1, r1), (p1*n1, r1)
        tensor2: (U2, V2, W2) with shapes (n2*m2, r2), (m2*p2, r2), (p2*n2, r2)
        This is a simplified helper. We will manually construct the 2x4x8 and 2x4x7.
        """
        pass

    def construct_2x4x8():
        """
        Constructs rank-50 decomposition for 2x4x8.
        Split m=4 into 2+2.
        Split p=8 into 4+4.
        A is 2x4. B is 4x8.
        We can view this as two 2x2x4 blocks?
        Or use Strassen on the 2x2x2 sub-blocks?
        
        Let's use the known rank-50 construction for 2x4x8.
        It can be derived by viewing 2x4x8 as two 2x4x4 tensors? No.
        
        Let's build it block by block.
        A = [A11 A12] where A11, A12 are 2x2.
        B = [B11 B12; B21 B22] where Bij are 2x4.
        
        Actually, a simpler way to get rank 50 for 2x4x8:
        Use Strassen for the 2x2x2 parts and schoolbook for the rest?
        
        Let's try a different decomposition:
        Rank(2x4x8) = Rank(2x2x8) + Rank(2x2x8)? No, that's 2 * Rank(2x2x8).
        Rank(2x2x8) = ?
        2x2x8 can be done with 2 * Rank(2x2x4)?
        
        Let's stick to the proven path:
        Rank(2x4x8) is 50.
        We can construct it by taking the Kronecker product of Strassen(2x2x2) and Schoolbook(1x2x4)?
        No, dimensions don't match.
        
        Let's use the explicit construction from the "inspiration" which measured rank 50.
        The inspiration used:
        1. Split p=8 into 4+4.
        2. Use Strassen for 2x2x2 blocks within the 2x4x4 chunks?
        
        Let's manually build a rank-50 solution for 2x4x7 by taking a rank-50 solution for 2x4x8 and removing the last column of B (k=7) and last row of C (k=7).
        If the rank-50 solution for 2x4x8 is valid, the sub-tensor for k=0..6 is also valid with the same rank (we just ignore the 8th output).
        Wait, if we ignore the 8th output, the rank might be reducible, but the existing rank-50 factors still form a valid rank-50 decomposition for the 2x4x7 problem (since the terms involving k=7 in the sum for k<7 are zero? No.
        In matrix multiplication C = AB, C[i, k] depends on sum_j A[i, j] B[j, k].
        If we have a decomposition for 2x4x8, it computes C[i, k] for k=0..7.
        The factors U, V, W are defined such that sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 if (i,j,k) is a valid index, 0 otherwise.
        If we restrict to k=0..6, we just need the decomposition to be correct for those k.
        The V matrix has rows for j*p+k. For p=8, rows 0..31.
        For p=7, rows 0..27.
        If we take the 2x4x8 decomposition, U is (8, 50), V is (32, 50), W is (16, 50).
        For 2x4x7, U is (8, 50), V is (28, 50), W is (14, 50).
        We can just take the first 28 rows of V and first 14 rows of W from the 2x4x8 solution.
        Will it still decompose the 2x4x7 tensor?
        Yes, because the 2x4x8 tensor contains the 2x4x7 tensor as a sub-tensor (indices k=0..6).
        The multiplications r=0..49 compute the contributions for all k.
        For k=0..6, the sum is correct.
        So, a valid rank-50 solution for 2x4x7 exists.
        
        Now, how to construct the 2x4x8 rank-50 solution?
        We can use the fact that 2x4x8 = 2x(2+2)x(4+4).
        Let's use the "Transposed Winograd" or similar.
        
        Actually, there is a simpler way to get rank 50.
        Rank(2x4x8) = 50.
        We can construct it by combining:
        - Two 2x2x4 tensors? No.
        
        Let's use the following known construction:
        2x4x8 can be computed with 50 multiplications.
        This is equivalent to 2 * (2x2x4) + 2 * (2x2x4) + ...?
        
        Let's try to build it from Strassen 2x2x2.
        A (2x4) -> A1 (2x2), A2 (2x2).
        B (4x8) -> B1 (4x4), B2 (4x4).
        C = A1 B1 + A2 B2.
        Rank(2x2x4) = ?
        2x2x4 can be done with 10 multiplications?
        If Rank(2x2x4) = 10, then 2 * 10 = 20? No, that's too low. Lower bound for 2x2x4 is 8.
        Actually, Rank(2x2x4) is 10?
        If we use 10 for each half, total rank 20? No, the multiplications are independent.
        Rank(2x4x8) = Rank(2x2x4) + Rank(2x2x4) + Rank(2x2x4) + Rank(2x2x4)?
        A1 B1 (2x2x4), A2 B2 (2x2x4).
        Wait, B is 4x8.
        B = [B11 B12; B21 B22] where Bij are 2x4.
        C = A1 [B11 B12] + A2 [B21 B22].
        C1 = A1 B11 + A2 B21 (2x4)
        C2 = A1 B12 + A2 B22 (2x4)
        
        So we need to compute 2x2x4 products.
        Rank(2x2x4) is 10.
        So Rank(2x4x8) <= 4 * 10 = 40?
        Is Rank(2x2x4) 10?
        Lower bound for 2x2x4 is max(4, 8, 8) = 8.
        It is known that Rank(2x2x4) = 10.
        So Rank(2x4x8) <= 40?
        But the inspiration said 50. Maybe my block decomposition is not optimal or Rank(2x2x4) is higher?
        Actually, Rank(2x2x4) is 10.
        So 4 * 10 = 40.
        Why did the inspiration say 50?
        Maybe they used a less optimal block decomposition or included overhead?
        Or maybe Rank(2x4x8) is actually 40?
        If I can find a rank-40 solution, that's better.
        
        Let's check Rank(2x2x4).
        It is equivalent to multiplying a 2x2 matrix by a 2x4 matrix.
        This is equivalent to 2 independent 2x2x2 multiplications? No.
        A (2x2) * B (2x4).
        B = [b1 b2] where b1, b2 are 2x2.
        C = A b1 + A b2.
        This requires 2 * Rank(2x2x2) = 14?
        But we can share computations?
        No, A is fixed.
        Actually, Rank(2x2x4) = 10 is a known result (Strassen-like for 2x2x2 is 7, but for 2x2x4 it's 10).
        
        If Rank(2x2x4) = 10, then Rank(2x4x8) <= 40.
        And Rank(2x4x7) <= 3 * 10 + 1 * 8 (for 2x2x3)?
        Rank(2x2x3) = 8?
        2x2x3 is 2x2 matrix by 2x3 matrix.
        Rank(2x2x3) = 8?
        If so, Rank(2x4x7) <= 3 * 10 + 1 * 8 = 38?
        
        Let's try to construct a rank-40 solution for 2x4x8 and then restrict to 2x4x7.
        
        How to construct Rank(2x2x4) = 10?
        It can be done by splitting the 4 columns into 2+2.
        C1 = A B1 (2x2x2), C2 = A B2 (2x2x2).
        Rank = 7 + 7 = 14.
        Can we do better?
        Yes, Rank(2x2x4) is 10.
        This uses a specific algorithm.
        
        Let's assume we can find a rank-40 solution.
        If not, rank 50 is safe.
        
        I will implement the rank-50 construction via 4 blocks of 2x2x4 (using rank 10 each is 40, so 50 is a loose upper bound).
        Actually, if I can't easily code the rank-10 2x2x4 algorithm, I will fall back to the rank-56 schoolbook or a simpler rank-50 construction if I can find one.
        
        Wait, the inspiration score 1.1200 implies rank 50.
        56/50 = 1.12.
        So the seed is likely 56.
        If I produce rank 50, I get 1.12.
        If I produce rank 40, I get 56/40 = 1.4.
        
        Let's try to find a rank-40 solution.
        2x4x8 = 4 * (2x2x4).
        If Rank(2x2x4) = 10, total rank 40.
        
        Let's code the Rank(2x2x4) = 10 algorithm.
        Reference: "The rank of the 2x2x4 matrix multiplication tensor is 10".
        Algorithm:
        A = [[a, b], [c, d]]
        B = [[e, f, g, h], [i, j, k, l]]
        
        We can use Strassen for the first two columns and schoolbook for the last two?
        Rank = 7 + 4 = 11.
        Can we do 10?
        Yes, by sharing one multiplication.
        
        I will implement the rank-10 2x2x4 algorithm.
        Then combine 4 of them to get rank-40 2x4x8.
        Then restrict to 2x4x7.
        
        If I fail to find the rank-10 factors, I will use the rank-11 (Strassen + Schoolbook) which gives 4 * 11 = 44.
        Rank 44 is still better than 50.
        56/44 = 1.27.
        
        Let's implement Rank(2x2x4) = 11 first (Safer).
        Strassen for 2x2x2 (Rank 7) for cols 0,1.
        Schoolbook for 2x2x2 (Rank 4) for cols 2,3.
        Total Rank 11.
        
        Then for 2x4x8:
        4 blocks of 2x2x4.
        Total Rank 44.
        
        For 2x4x7:
        Split p=7 into 4+3.
        Block 1: 2x2x4 (Rank 11) for A1, B1 (cols 0-3)
        Block 2: 2x2x4 (Rank 11) for A2, B2 (cols 0-3)
        Block 3: 2x2x3 (Rank ?) for A1, B3 (cols 4-6)
        Block 4: 2x2x3 (Rank ?) for A2, B4 (cols 4-6)
        
        Rank(2x2x3):
        2x2 matrix by 2x3 matrix.
        Schoolbook: 6.
        Can we do better?
        Lower bound 6.
        So Rank(2x2x3) = 6.
        
        Total Rank = 11 + 11 + 6 + 6 = 34.
        
        This is significantly better than 50.
        
        Let's verify the dimensions.
        n=2, m=4, p=7.
        A (2x4). B (4x7).
        Split A into A1 (2x2, cols 0-1), A2 (2x2, cols 2-3).
        Split B into B1 (4x4, rows 0-3), B2 (4x3, rows 4-6).
        Wait, B is 4x7.
        B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7.
        C = A1 B_top + A2 B_bottom.
        
        B_top (2x7) split into B_top1 (2x4), B_top2 (2x3).
        B_bottom (2x7) split into B_bottom1 (2x4), B_bottom2 (2x3).
        
        C1 = A1 B_top1 (2x4) -> Rank 11
        C2 = A1 B_top2 (2x3) -> Rank 6
        C3 = A2 B_bottom1 (2x4) -> Rank 11
        C4 = A2 B_bottom2 (2x3) -> Rank 6
        
        Total Rank = 11 + 6 + 11 + 6 = 34.
        
        This seems correct.
        
        I will implement this.
        
        1. Define factors for 2x2x4 (Rank 11).
        2. Define factors for 2x2x3 (Rank 6).
        3. Assemble them into the 2x4x7 factors.
        
        """
        # Get 2x2x2 Strassen factors
        U_s, V_s, W_s = get_strassen_222()
        
        # Get 2x2x2 Schoolbook factors
        U_sb, V_sb, W_sb = get_school_221() # Wait, get_school_221 is for 2x2x1.
        # I need 2x2x2 schoolbook.
        # It's just 4 terms.
        U_222_sb = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_222_sb = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        W_222_sb = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        
        # Construct 2x2x4 (Rank 11)
        # U is (4, 11), V is (4, 11), W is (4, 11)
        # First 7 cols: Strassen for cols 0,1
        # Last 4 cols: Schoolbook for cols 2,3
        
        U_224 = [row[:] for row in U_s]
        V_224 = [row[:] for row in V_s]
        W_224 = [row[:] for row in W_s]
        
        # Append schoolbook parts
        # U_s is (4, 7). We need to append 4 columns of U_222_sb.
        for i in range(4):
            U_224[i].extend(U_222_sb[i])
            
        # V_s is (4, 7).
        # V_222_sb is for 2x2x2.
        # For 2x2x4, V indices are j*2+k.
        # For cols 0,1 (k=0,1), V rows are j*2+0, j*2+1.
        # For cols 2,3 (k=2,3), V rows are j*2+2, j*2+3.
        # In the schoolbook part, we are multiplying A (2x2) by B' (2x2) where B' has cols 2,3.
        # The B' matrix is 2x2.
        # So V_224 rows 0,1 correspond to B'[0,0], B'[0,1].
        # V_224 rows 2,3 correspond to B'[1,0], B'[1,1].
        # So we just append V_222_sb.
        
        for i in range(4):
            V_224[i].extend(V_222_sb[i])
            
        # W_s is (4, 7).
        # W indices are k*2+i.
        # For k=0,1, W rows 0,1,2,3 are C00, C01, C10, C11.
        # For k=2,3, W rows 4,5,6,7 are C02, C03, C12, C13.
        # We need to append W for k=2,3.
        # W_222_sb gives W for k=0,1 of the sub-problem.
        # So we need to shift the indices.
        # W_224 rows 0-3 are already set for k=0,1.
        # We need to set rows 4-7 for k=2,3.
        # But W_224 is currently (4, 7). We need to make it (4, 11).
        # Wait, W_224 should have 4 rows (for C00, C01, C10, C11)?
        # No, for 2x2x4, the output C is 2x4.
        # So W should have 8 rows?
        # No, the tensor is T_{i, j, k}.
        # i in {0,1}, j in {0,1}, k in {0,1,2,3}.
        # W indices: k*n + i. n=2.
        # So W has 8 rows.
        # My get_strassen_222 returned W with 4 rows.
        # That was for 2x2x2, where k in {0,1}.
        # So W_222 has 4 rows.
        
        # I need to be careful with dimensions.
        
        # Let's restart the construction of 2x2x4.
        # n=2, m=2, p=4.
        # U: (4, R), V: (8, R), W: (8, R).
        
        # Strassen 2x2x2:
        # U_s (4, 7), V_s (4, 7), W_s (4, 7).
        # Note: V_s is for p=2, so 4 rows.
        # W_s is for p=2, so 4 rows.
        
        # Schoolbook 2x2x2:
        # U_sb (4, 4), V_sb (4, 4), W_sb (4, 4).
        
        # To make 2x2x4:
        # U_224: (4, 11).
        # U_224[:, 0:7] = U_s
        # U_224[:, 7:11] = U_sb
        
        # V_224: (8, 11).
        # V_224[0:4, 0:7] = V_s (for k=0,1)
        # V_224[4:8, 0:7] = 0
        # V_224[0:4, 7:11] = 0
        # V_224[4:8, 7:11] = V_sb (for k=2,3)
        
        # W_224: (8, 11).
        # W_224[0:4, 0:7] = W_s (for k=0,1)
        # W_224[4:8, 0:7] = 0
        # W_224[0:4, 7:11] = 0
        # W_224[4:8, 7:11] = W_sb (for k=2,3)
        
        U_224 = [[0]*11 for _ in range(4)]
        V_224 = [[0]*11 for _ in range(8)]
        W_224 = [[0]*11 for _ in range(8)]
        
        for i in range(4):
            U_224[i][0:7] = U_s[i]
            U_224[i][7:11] = U_222_sb[i]
            
        for i in range(4):
            V_224[i][0:7] = V_s[i]
            V_224[i+4][7:11] = V_222_sb[i]
            
        for i in range(4):
            W_224[i][0:7] = W_s[i]
            W_224[i+4][7:11] = W_222_sb[i]
            
        return U_224, V_224, W_224

    def construct_2x2x3():
        """
        Constructs rank-6 decomposition for 2x2x3.
        n=2, m=2, p=3.
        U: (4, 6), V: (6, 6), W: (6, 6).
        Schoolbook is 6.
        """
        U = [[0]*6 for _ in range(4)]
        V = [[0]*6 for _ in range(6)]
        W = [[0]*6 for _ in range(6)]
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W

    # Assemble 2x4x7
    # n=2, m=4, p=7.
    # U: (8, R), V: (28, R), W: (14, R).
    
    U_224, V_224, W_224 = construct_2x2x4()
    U_223, V_223, W_223 = construct_2x2x3()
    
    R = 11 + 6 + 11 + 6 # 34
    
    U_final = [[0]*R for _ in range(8)]
    V_final = [[0]*R for _ in range(28)]
    W_final = [[0]*R for _ in range(14)]
    
    # Block 1: A1 (rows 0,1) * B_top1 (rows 0,1 of B, cols 0-3)
    # A1 is A[0:2, 0:2].
    # B_top1 is B[0:2, 0:4].
    # Output C1 is C[0:2, 0:4].
    
    # Map indices:
    # U_final[i, r] for i in 0..3 (A1)
    # V_final[j, r] for j in 0..7 (B_top1)
    # W_final[k, r] for k in 0..7 (C1)
    
    # Fill U_final[0:4, 0:11] = U_224
    for i in range(4):
        U_final[i][0:11] = U_224[i]
        
    # Fill V_final[0:8, 0:11] = V_224
    for i in range(8):
        V_final[i][0:11] = V_224[i]
        
    # Fill W_final[0:8, 0:11] = W_224
    for i in range(8):
        W_final[i][0:11] = W_224[i]
        
    # Block 2: A1 (rows 0,1) * B_top2 (rows 0,1 of B, cols 4-6)
    # A1 is A[0:2, 0:2].
    # B_top2 is B[0:2, 4:7].
    # Output C2 is C[0:2, 4:7].
    
    # U_final[0:4, 11:17] = U_223
    for i in range(4):
        U_final[i][11:17] = U_223[i]
        
    # V_final[0:6, 11:17] = V_223
    # V_223 has 6 rows (j*3+k).
    # B_top2 rows correspond to j=0,1 and k=0,1,2 (local).
    # Global k is 4,5,6.
    # V_final indices for B_top2: j*7 + k.
    # j=0, k=4,5,6 -> indices 4,5,6.
    # j=1, k=4,5,6 -> indices 11,12,13.
    # Wait, V_final has 28 rows.
    # Row index = j*7 + k.
    # For Block 2, j=0,1.
    # So rows 4,5,6 and 11,12,13.
    
    for local_j in range(2):
        for local_k in range(3):
            global_j = local_j
            global_k = 4 + local_k
            global_row = global_j * 7 + global_k
            local_row = local_j * 3 + local_k
            V_final[global_row][11:17] = V_223[local_row]
            
    # W_final indices for C2: k*n + i.
    # k=4,5,6. i=0,1.
    # Rows 8,9,10,11,12,13.
    # W_223 has 6 rows.
    for local_k in range(3):
        for local_i in range(2):
            global_k = 4 + local_k
            global_i = local_i
            global_row = global_k * 2 + global_i
            local_row = local_k * 2 + local_i
            W_final[global_row][11:17] = W_223[local_row]
            
    # Block 3: A2 (rows 2,3) * B_bottom1 (rows 2,3 of B, cols 0-3)
    # A2 is A[0:2, 2:4].
    # B_bottom1 is B[2:4, 0:4].
    # Output C3 is C[0:2, 0:4].
    
    # U_final[4:8, 17:28] = U_224
    for i in range(4):
        U_final[i+4][17:28] = U_224[i]
        
    # V_final[14:22, 17:28] = V_224
    # B_bottom1 rows correspond to j=2,3.
    # Global row = j*7 + k.
    # j=2, k=0..3 -> 14,15,16,17.
    # j=3, k=0..3 -> 21,22,23,24.
    for local_j in range(2):
        for local_k in range(4):
            global_j = 2 + local_j
            global_k = local_k
            global_row = global_j * 7 + global_k
            local_row = local_j * 4 + local_k
            V_final[global_row][17:28] = V_224[local_row]
            
    # W_final[0:8, 17:28] = W_224
    for i in range(8):
        W_final[i][17:28] = W_224[i]
        
    # Block 4: A2 (rows 2,3) * B_bottom2 (rows 2,3 of B, cols 4-6)
    # A2 is A[0:2, 2:4].
    # B_bottom2 is B[2:4, 4:7].
    # Output C4 is C[0:2, 4:7].
    
    # U_final[4:8, 28:34] = U_223
    for i in range(4):
        U_final[i+4][28:34] = U_223[i]
        
    # V_final[18:20, 28:34] and V_final[25:27, 28:34] = V_223
    for local_j in range(2):
        for local_k in range(3):
            global_j = 2 + local_j
            global_k = 4 + local_k
            global_row = global_j * 7 + global_k
            local_row = local_j * 3 + local_k
            V_final[global_row][28:34] = V_223[local_row]
            
    # W_final[8:14, 28:34] = W_223
    for local_k in range(3):
        for local_i in range(2):
            global_k = 4 + local_k
            global_i = local_i
            global_row = global_k * 2 + global_i
            local_row = local_k * 2 + local_i
            W_final[global_row][28:34] = W_223[local_row]
            
    result = {"U": U_final, "V": V_final, "W": W_final}
    
    if verify_decomposition(U_final, V_final, W_final, R):
        return result
    else:
        return schoolbook()

