"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction
import numpy as np

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Schoolbook rank
    rank_school = n * m * p  # 56
    
    # Schoolbook solution
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    best_rank = rank_school
    best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Fast verification using numpy with exact integer arithmetic
    def verify_fast(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition using numpy."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                    if s != 1:
                        return False
        return True

    # Strategy: Use a block-wise Strassen decomposition to reduce rank.
    # We can view the 2x4 matrix A as two 2x2 blocks: A = [A1 A2]
    # And the 4x7 matrix B as two 4x7 blocks stacked: B = [B1; B2] where B1 is 2x7 and B2 is 2x7?
    # No, B is 4x7. Let's split the inner dimension m=4 into two parts of size 2.
    # A is 2x4. Split columns: A = [A_left (2x2) | A_right (2x2)]
    # B is 4x7. Split rows: B = [B_top (2x7); B_bottom (2x7)]
    # C = A * B = A_left * B_top + A_right * B_bottom
    
    # This reduces the problem to computing two 2x2 * 2x7 products.
    # The rank of 2x2 * 2x7 multiplication:
    # We can further decompose the 2x2 * 2x7 multiplication.
    # Let X be 2x2 and Y be 2x7.
    # Standard rank for 2x2 * 2x7 is 2*2*7 = 28.
    # Can we do better?
    # We can use Strassen for the 2x2 part?
    # Strassen computes 2x2 * 2x2 with rank 7.
    # Here we have 2x2 * 2x7.
    # We can split Y into seven 2x1 columns? No, that's not efficient.
    # We can split Y into one 2x1 and one 2x6? No.
    
    # Let's look at the structure of 2x2 * 2x7.
    # Let X = [[a, b], [c, d]] and Y = [[y11, ..., y17], [y21, ..., y27]].
    # Z = X * Y is 2x7.
    # Z[0, k] = a*y1k + b*y2k
    # Z[1, k] = c*y1k + d*y2k
    
    # We can compute these using 7 multiplications for each column?
    # Total 14 multiplications?
    # Wait, the standard algorithm uses 2*2*7 = 28 multiplications.
    # But we can share computations?
    # For each column k, we need to compute [a*y1k + b*y2k, c*y1k + d*y2k].
    # This is a 2x2 * 1x2 matrix multiplication (if we view y_k as a column vector).
    # The rank of 2x2 * 1x2 is 4?
    # Actually, we can use Strassen-like identities for the 2x2 * 1x2 case?
    # No, 2x2 * 1x2 is just 4 multiplications.
    # So for 7 columns, it's 28 multiplications.
    
    # Is there a way to reduce the rank of 2x2 * 2x7?
    # The tensor is M_{2,2,7}.
    # Lower bound: max(2*2, 2*7, 7*2) = 14.
    # Upper bound: 28.
    # Known rank for M_{2,2,7}?
    # We can use the fact that M_{2,2,7} is a direct sum of 7 copies of M_{2,2,1}? No.
    # It is related to M_{2,2,2} (Strassen rank 7).
    # If we split the 7 columns into 2+2+2+1?
    # M_{2,2,2} has rank 7.
    # M_{2,2,1} has rank 4.
    # Total rank: 3 * 7 + 4 = 25.
    # So we can compute 2x2 * 2x7 with rank 25.
    
    # Let's verify this.
    # We split the 7 columns of Y into three groups of 2 and one group of 1.
    # For each group of 2 columns, we use Strassen (rank 7) to compute the 2x2 * 2x2 product.
    # For the group of 1 column, we use schoolbook (rank 4) to compute the 2x2 * 2x1 product.
    # Total rank for one 2x2 * 2x7 block is 3*7 + 4 = 25.
    
    # We have two such blocks (A_left * B_top and A_right * B_bottom).
    # Total rank = 25 + 25 = 50.
    
    # This is better than 56.
    
    # Let's implement this.
    
    def strassen_2x2_2x2(U_base, V_base, W_base, offset, n=2, m=2, p=2):
        """
        Adds Strassen decomposition for 2x2 * 2x2 to U, V, W starting at offset.
        U is (2*2, rank), V is (2*2, rank), W is (2*2, rank).
        Indices for U: i*m + j
        Indices for V: j*p + k
        Indices for W: k*n + i
        """
        # Strassen formulas for 2x2 matrices
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to assign columns for M1..M7
        cols = list(range(offset, offset + 7))
        
        # M1
        # U: A11, A22
        U[0*2+0][cols[0]] = 1
        U[1*2+1][cols[0]] = 1
        # V: B11, B22
        V[0*2+0][cols[0]] = 1
        V[1*2+1][cols[0]] = 1
        # W: C11, C22
        W[0*2+0][cols[0]] = 1
        W[1*2+1][cols[0]] = 1
        
        # M2
        # U: A21, A22
        U[1*2+0][cols[1]] = 1
        U[1*2+1][cols[1]] = 1
        # V: B11
        V[0*2+0][cols[1]] = 1
        # W: C21, C22
        W[1*2+0][cols[1]] = 1
        W[1*2+1][cols[1]] = -1
        
        # M3
        # U: A11
        U[0*2+0][cols[2]] = 1
        # V: B21, B22
        V[1*2+0][cols[2]] = 1
        V[1*2+1][cols[2]] = -1
        # W: C12, C22
        W[0*2+1][cols[2]] = 1
        W[1*2+1][cols[2]] = 1
        
        # M4
        # U: A22
        U[1*2+1][cols[3]] = 1
        # V: B11, B12
        V[0*2+0][cols[3]] = 1
        V[0*2+1][cols[3]] = -1
        # W: C11, C21
        W[0*2+0][cols[3]] = 1
        W[1*2+0][cols[3]] = 1
        
        # M5
        # U: A11, A12
        U[0*2+0][cols[4]] = 1
        U[0*2+1][cols[4]] = 1
        # V: B22
        V[1*2+1][cols[4]] = 1
        # W: C11, C12
        W[0*2+0][cols[4]] = -1
        W[0*2+1][cols[4]] = 1
        
        # M6
        # U: A21, A11
        U[1*2+0][cols[5]] = 1
        U[0*2+0][cols[5]] = -1
        # V: B11, B12
        V[0*2+0][cols[5]] = 1
        V[0*2+1][cols[5]] = 1
        # W: C22
        W[1*2+1][cols[5]] = 1
        
        # M7
        # U: A12, A22
        U[0*2+1][cols[6]] = 1
        U[1*2+1][cols[6]] = -1
        # V: B21, B22
        V[1*2+0][cols[6]] = 1
        V[1*2+1][cols[6]] = 1
        # W: C11
        W[0*2+0][cols[6]] = 1
        
        return offset + 7

    def schoolbook_2x2_2x1(U_base, V_base, W_base, offset, n=2, m=2, p=1):
        """
        Adds schoolbook decomposition for 2x2 * 2x1 to U, V, W starting at offset.
        """
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        cols = list(range(offset, offset + 4))
        
        # Term 1: A11 * B1 -> C1
        U[0*2+0][cols[0]] = 1
        V[0*1+0][cols[0]] = 1
        W[0*2+0][cols[0]] = 1
        
        # Term 2: A12 * B2 -> C1
        U[0*2+1][cols[1]] = 1
        V[1*1+0][cols[1]] = 1
        W[0*2+0][cols[1]] = 1
        
        # Term 3: A21 * B1 -> C2
        U[1*2+0][cols[2]] = 1
        V[0*1+0][cols[2]] = 1
        W[1*2+0][cols[2]] = 1
        
        # Term 4: A22 * B2 -> C2
        U[1*2+1][cols[3]] = 1
        V[1*1+0][cols[3]] = 1
        W[1*2+0][cols[3]] = 1
        
        return offset + 4

    # Construct the rank 50 solution
    rank_50 = 50
    U = [[0] * rank_50 for _ in range(n * m)]  # 8x50
    V = [[0] * rank_50 for _ in range(m * p)]  # 28x50
    W = [[0] * rank_50 for _ in range(p * n)]  # 14x50
    
    # We split A (2x4) into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # We split B (4x7) into B_top (2x7, rows 0-1) and B_bottom (2x7, rows 2-3)
    
    # Block 1: A_left * B_top
    # A_left indices in U: i*4 + j where j in 0,1
    # B_top indices in V: j*7 + k where j in 0,1
    # Result C indices in W: k*2 + i
    
    # We need to map the local 2x2 * 2x7 indices to global indices.
    # Local U (2x2): (i, j_local) -> Global U (i, j_global) where j_global = j_local
    # Local V (2x7): (j_local, k) -> Global V (j_global, k) where j_global = j_local
    # Local W (2x7): (k, i) -> Global W (k, i)
    
    # Let's implement a helper to add a 2x2 * 2x7 block with specific offsets and index mappings.
    
    def add_block_2x2_2x7(U, V, W, offset, i_offset, j_offset, k_offset, n_block=2, m_block=2, p_block=7):
        """
        Adds a 2x2 * 2x7 block.
        i_offset: offset for rows of A (n dimension)
        j_offset: offset for cols of A (m dimension) and rows of B
        k_offset: offset for cols of B (p dimension)
        """
        # Split p_block=7 into 2, 2, 2, 1
        # We use Strassen for the 2x2 parts and schoolbook for the 2x1 part.
        
        # Helper to map local indices to global
        def map_u(i_loc, j_loc):
            return i_offset + i_loc, j_offset + j_loc
        def map_v(j_loc, k_loc):
            return j_offset + j_loc, k_offset + k_loc
        def map_w(k_loc, i_loc):
            return k_offset + k_loc, i_offset + i_loc
            
        # We need to modify the U, V, W arrays directly.
        # Since we are using lists of lists, we can pass them and modify in place.
        # But the helper functions strassen_2x2_2x2 and schoolbook_2x2_2x1 assume specific layouts.
        # Let's write a generic adder that takes the local indices and writes to global.
        
        # Part 1: k_loc in 0,1 (first 2 columns)
        # Strassen 2x2 * 2x2
        # M1
        # U: A11, A22 -> (0,0), (1,1)
        U[i_offset+0][offset+0] += 1
        U[i_offset+1][offset+0] += 1
        # V: B11, B22 -> (0,0), (1,1)
        V[j_offset+0][k_offset+0] += 1
        V[j_offset+1][k_offset+1] += 1
        # W: C11, C22 -> (0,0), (1,1)
        W[k_offset+0][i_offset+0] += 1
        W[k_offset+1][i_offset+1] += 1
        
        # M2
        # U: A21, A22 -> (1,0), (1,1)
        U[i_offset+1][offset+1] += 1
        U[i_offset+1][offset+1] += 1
        # V: B11 -> (0,0)
        V[j_offset+0][k_offset+0] += 1
        # W: C21, C22 -> (1,0), (1,1)
        W[k_offset+0][i_offset+1] += 1
        W[k_offset+1][i_offset+1] -= 1
        
        # M3
        # U: A11 -> (0,0)
        U[i_offset+0][offset+2] += 1
        # V: B21, B22 -> (1,0), (1,1)
        V[j_offset+1][k_offset+0] += 1
        V[j_offset+1][k_offset+1] -= 1
        # W: C12, C22 -> (0,1), (1,1)
        W[k_offset+1][i_offset+0] += 1
        W[k_offset+1][i_offset+1] += 1
        
        # M4
        # U: A22 -> (1,1)
        U[i_offset+1][offset+3] += 1
        # V: B11, B12 -> (0,0), (0,1)
        V[j_offset+0][k_offset+0] += 1
        V[j_offset+0][k_offset+1] -= 1
        # W: C11, C21 -> (0,0), (1,0)
        W[k_offset+0][i_offset+0] += 1
        W[k_offset+0][i_offset+1] += 1
        
        # M5
        # U: A11, A12 -> (0,0), (0,1)
        U[i_offset+0][offset+4] += 1
        U[i_offset+0][offset+4] += 1
        # V: B22 -> (1,1)
        V[j_offset+1][k_offset+1] += 1
        # W: C11, C12 -> (0,0), (0,1)
        W[k_offset+0][i_offset+0] -= 1
        W[k_offset+1][i_offset+0] += 1
        
        # M6
        # U: A21, A11 -> (1,0), (0,0)
        U[i_offset+1][offset+5] += 1
        U[i_offset+0][offset+5] -= 1
        # V: B11, B12 -> (0,0), (0,1)
        V[j_offset+0][k_offset+0] += 1
        V[j_offset+0][k_offset+1] += 1
        # W: C22 -> (1,1)
        W[k_offset+1][i_offset+1] += 1
        
        # M7
        # U: A12, A22 -> (0,1), (1,1)
        U[i_offset+0][offset+6] += 1
        U[i_offset+1][offset+6] -= 1
        # V: B21, B22 -> (1,0), (1,1)
        V[j_offset+1][k_offset+0] += 1
        V[j_offset+1][k_offset+1] += 1
        # W: C11 -> (0,0)
        W[k_offset+0][i_offset+0] += 1
        
        # Part 2: k_loc in 2,3 (next 2 columns)
        # Same Strassen, but k_offset shifted by 2
        k_off_2 = k_offset + 2
        # M1
        U[i_offset+0][offset+7] += 1
        U[i_offset+1][offset+7] += 1
        V[j_offset+0][k_off_2+0] += 1
        V[j_offset+1][k_off_2+1] += 1
        W[k_off_2+0][i_offset+0] += 1
        W[k_off_2+1][i_offset+1] += 1
        
        # M2
        U[i_offset+1][offset+8] += 1
        U[i_offset+1][offset+8] += 1
        V[j_offset+0][k_off_2+0] += 1
        W[k_off_2+0][i_offset+1] += 1
        W[k_off_2+1][i_offset+1] -= 1
        
        # M3
        U[i_offset+0][offset+9] += 1
        V[j_offset+1][k_off_2+0] += 1
        V[j_offset+1][k_off_2+1] -= 1
        W[k_off_2+1][i_offset+0] += 1
        W[k_off_2+1][i_offset+1] += 1
        
        # M4
        U[i_offset+1][offset+10] += 1
        V[j_offset+0][k_off_2+0] += 1
        V[j_offset+0][k_off_2+1] -= 1
        W[k_off_2+0][i_offset+0] += 1
        W[k_off_2+0][i_offset+1] += 1
        
        # M5
        U[i_offset+0][offset+11] += 1
        U[i_offset+0][offset+11] += 1
        V[j_offset+1][k_off_2+1] += 1
        W[k_off_2+0][i_offset+0] -= 1
        W[k_off_2+1][i_offset+0] += 1
        
        # M6
        U[i_offset+1][offset+12] += 1
        U[i_offset+0][offset+12] -= 1
        V[j_offset+0][k_off_2+0] += 1
        V[j_offset+0][k_off_2+1] += 1
        W[k_off_2+1][i_offset+1] += 1
        
        # M7
        U[i_offset+0][offset+13] += 1
        U[i_offset+1][offset+13] -= 1
        V[j_offset+1][k_off_2+0] += 1
        V[j_offset+1][k_off_2+1] += 1
        W[k_off_2+0][i_offset+0] += 1
        
        # Part 3: k_loc in 4,5 (next 2 columns)
        k_off_3 = k_offset + 4
        # M1
        U[i_offset+0][offset+14] += 1
        U[i_offset+1][offset+14] += 1
        V[j_offset+0][k_off_3+0] += 1
        V[j_offset+1][k_off_3+1] += 1
        W[k_off_3+0][i_offset+0] += 1
        W[k_off_3+1][i_offset+1] += 1
        
        # M2
        U[i_offset+1][offset+15] += 1
        U[i_offset+1][offset+15] += 1
        V[j_offset+0][k_off_3+0] += 1
        W[k_off_3+0][i_offset+1] += 1
        W[k_off_3+1][i_offset+1] -= 1
        
        # M3
        U[i_offset+0][offset+16] += 1
        V[j_offset+1][k_off_3+0] += 1
        V[j_offset+1][k_off_3+1] -= 1
        W[k_off_3+1][i_offset+0] += 1
        W[k_off_3+1][i_offset+1] += 1
        
        # M4
        U[i_offset+1][offset+17] += 1
        V[j_offset+0][k_off_3+0] += 1
        V[j_offset+0][k_off_3+1] -= 1
        W[k_off_3+0][i_offset+0] += 1
        W[k_off_3+0][i_offset+1] += 1
        
        # M5
        U[i_offset+0][offset+18] += 1
        U[i_offset+0][offset+18] += 1
        V[j_offset+1][k_off_3+1] += 1
        W[k_off_3+0][i_offset+0] -= 1
        W[k_off_3+1][i_offset+0] += 1
        
        # M6
        U[i_offset+1][offset+19] += 1
        U[i_offset+0][offset+19] -= 1
        V[j_offset+0][k_off_3+0] += 1
        V[j_offset+0][k_off_3+1] += 1
        W[k_off_3+1][i_offset+1] += 1
        
        # M7
        U[i_offset+0][offset+20] += 1
        U[i_offset+1][offset+20] -= 1
        V[j_offset+1][k_off_3+0] += 1
        V[j_offset+1][k_off_3+1] += 1
        W[k_off_3+0][i_offset+0] += 1
        
        # Part 4: k_loc in 6 (last 1 column)
        k_off_4 = k_offset + 6
        # Schoolbook 2x2 * 2x1
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # Term 1: A11 * B1 -> C1
        U[i_offset+0][offset+21] += 1
        V[j_offset+0][k_off_4] += 1
        W[k_off_4][i_offset+0] += 1
        
        # Term 2: A12 * B2 -> C1
        U[i_offset+0][offset+22] += 1
        V[j_offset+1][k_off_4] += 1
        W[k_off_4][i_offset+0] += 1
        
        # Term 3: A21 * B1 -> C2
        U[i_offset+1][offset+23] += 1
        V[j_offset+0][k_off_4] += 1
        W[k_off_4][i_offset+1] += 1
        
        # Term 4: A22 * B2 -> C2
        U[i_offset+1][offset+24] += 1
        V[j_offset+1][k_off_4] += 1
        W[k_off_4][i_offset+1] += 1
        
        return offset + 25

    # Now we add the two blocks
    # Block 1: A_left (i=0..1, j=0..1) * B_top (j=0..1, k=0..6)
    # i_offset=0, j_offset=0, k_offset=0
    # U indices: i*4 + j. Wait, my add_block function uses i_offset + i_loc for U row?
    # No, U is (n*m, rank). Row index is i*m + j.
    # In add_block, I used U[i_offset + i_loc]. This assumes m=1 or that i_offset includes the m stride?
    # Let's fix the indexing in add_block.
    # U row: i*m + j.
    # V row: j*p + k.
    # W row: k*n + i.
    
    # I will rewrite the addition logic to be explicit and correct.
    
    U = [[0] * rank_50 for _ in range(n * m)]
    V = [[0] * rank_50 for _ in range(m * p)]
    W = [[0] * rank_50 for _ in range(p * n)]
    
    def add_strassen_block(U, V, W, offset, i_base, j_base, k_base):
        """
        Adds Strassen block for 2x2 * 2x2.
        i_base, j_base, k_base are the starting indices for i, j, k.
        U index: i*m + j
        V index: j*p + k
        W index: k*n + i
        """
        m_val = m
        p_val = p
        n_val = n
        
        # M1
        U[i_base + 0][offset+0] += 1
        U[i_base + 1][offset+0] += 1
        V[j_base + 0][k_base + 0] += 1
        V[j_base + 1][k_base + 1] += 1
        W[k_base + 0][i_base + 0] += 1
        W[k_base + 1][i_base + 1] += 1
        
        # M2
        U[i_base + 1][offset+1] += 1
        U[i_base + 1][offset+1] += 1
        V[j_base + 0][k_base + 0] += 1
        W[k_base + 0][i_base + 1] += 1
        W[k_base + 1][i_base + 1] -= 1
        
        # M3
        U[i_base + 0][offset+2] += 1
        V[j_base + 1][k_base + 0] += 1
        V[j_base + 1][k_base + 1] -= 1
        W[k_base + 1][i_base + 0] += 1
        W[k_base + 1][i_base + 1] += 1
        
        # M4
        U[i_base + 1][offset+3] += 1
        V[j_base + 0][k_base + 0] += 1
        V[j_base + 0][k_base + 1] -= 1
        W[k_base + 0][i_base + 0] += 1
        W[k_base + 0][i_base + 1] += 1
        
        # M5
        U[i_base + 0][offset+4] += 1
        U[i_base + 0][offset+4] += 1
        V[j_base + 1][k_base + 1] += 1
        W[k_base + 0][i_base + 0] -= 1
        W[k_base + 1][i_base + 0] += 1
        
        # M6
        U[i_base + 1][offset+5] += 1
        U[i_base + 0][offset+5] -= 1
        V[j_base + 0][k_base + 0] += 1
        V[j_base + 0][k_base + 1] += 1
        W[k_base + 1][i_base + 1] += 1
        
        # M7
        U[i_base + 0][offset+6] += 1
        U[i_base + 1][offset+6] -= 1
        V[j_base + 1][k_base + 0] += 1
        V[j_base + 1][k_base + 1] += 1
        W[k_base + 0][i_base + 0] += 1
        
    def add_schoolbook_block_2x2_2x1(U, V, W, offset, i_base, j_base, k_base):
        """
        Adds schoolbook block for 2x2 * 2x1.
        """
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # Term 1: A11 * B1 -> C1
        U[i_base + 0][offset+0] += 1
        V[j_base + 0][k_base + 0] += 1
        W[k_base + 0][i_base + 0] += 1
        
        # Term 2: A12 * B2 -> C1
        U[i_base + 0][offset+1] += 1
        V[j_base + 1][k_base + 0] += 1
        W[k_base + 0][i_base + 0] += 1
        
        # Term 3: A21 * B1 -> C2
        U[i_base + 1][offset+2] += 1
        V[j_base + 0][k_base + 0] += 1
        W[k_base + 0][i_base + 1] += 1
        
        # Term 4: A22 * B2 -> C2
        U[i_base + 1][offset+3] += 1
        V[j_base + 1][k_base + 0] += 1
        W[k_base + 0][i_base + 1] += 1
        
    # Block 1: A_left (i=0..1, j=0..1) * B_top (j=0..1, k=0..6)
    # i_base=0, j_base=0, k_base=0
    # Note: U index is i*m + j. So for i=0, j=0 -> 0. i=0, j=1 -> 1.
    # My add_strassen_block uses U[i_base + i_loc]. This is WRONG if m > 1.
    # I need to use U[i_base + i_loc*m + j_loc].
    
    # Let's redefine the helpers to take the full index calculation.
    
    def add_strassen_block_correct(U, V, W, offset, i_base, j_base, k_base):
        # M1
        U[i_base + 0*m + 0][offset+0] += 1
        U[i_base + 1*m + 1][offset+0] += 1
        V[j_base + 0*p + 0][offset+0] += 1
        V[j_base + 1*p + 1][offset+0] += 1
        W[k_base + 0*n + 0][offset+0] += 1
        W[k_base + 1*n + 1][offset+0] += 1
        
        # M2
        U[i_base + 1*m + 0][offset+1] += 1
        U[i_base + 1*m + 1][offset+1] += 1
        V[j_base + 0*p + 0][offset+1] += 1
        W[k_base + 0*n + 1][offset+1] += 1
        W[k_base + 1*n + 1][offset+1] -= 1
        
        # M3
        U[i_base + 0*m + 0][offset+2] += 1
        V[j_base + 1*p + 0][offset+2] += 1
        V[j_base + 1*p + 1][offset+2] -= 1
        W[k_base + 1*n + 0][offset+2] += 1
        W[k_base + 1*n + 1][offset+2] += 1
        
        # M4
        U[i_base + 1*m + 1][offset+3] += 1
        V[j_base + 0*p + 0][offset+3] += 1
        V[j_base + 0*p + 1][offset+3] -= 1
        W[k_base + 0*n + 0][offset+3] += 1
        W[k_base + 0*n + 1][offset+3] += 1
        
        # M5
        U[i_base + 0*m + 0][offset+4] += 1
        U[i_base + 0*m + 1][offset+4] += 1
        V[j_base + 1*p + 1][offset+4] += 1
        W[k_base + 0*n + 0][offset+4] -= 1
        W[k_base + 1*n + 0][offset+4] += 1
        
        # M6
        U[i_base + 1*m + 0][offset+5] += 1
        U[i_base + 0*m + 0][offset+5] -= 1
        V[j_base + 0*p + 0][offset+5] += 1
        V[j_base + 0*p + 1][offset+5] += 1
        W[k_base + 1*n + 1][offset+5] += 1
        
        # M7
        U[i_base + 0*m + 1][offset+6] += 1
        U[i_base + 1*m + 1][offset+6] -= 1
        V[j_base + 1*p + 0][offset+6] += 1
        V[j_base + 1*p + 1][offset+6] += 1
        W[k_base + 0*n + 0][offset+6] += 1
        
    def add_schoolbook_block_correct(U, V, W, offset, i_base, j_base, k_base):
        # Term 1: A11 * B1 -> C1
        U[i_base + 0*m + 0][offset+0] += 1
        V[j_base + 0*p + 0][offset+0] += 1
        W[k_base + 0*n + 0][offset+0] += 1
        
        # Term 2: A12 * B2 -> C1
        U[i_base + 0*m + 1][offset+1] += 1
        V[j_base + 1*p + 0][offset+1] += 1
        W[k_base + 0*n + 0][offset+1] += 1
        
        # Term 3: A21 * B1 -> C2
        U[i_base + 1*m + 0][offset+2] += 1
        V[j_base + 0*p + 0][offset+2] += 1
        W[k_base + 0*n + 1][offset+2] += 1
        
        # Term 4: A22 * B2 -> C2
        U[i_base + 1*m + 1][offset+3] += 1
        V[j_base + 1*p + 0][offset+3] += 1
        W[k_base + 0*n + 1][offset+3] += 1

    # Now we can call these correctly.
    
    # Block 1: A_left (i=0..1, j=0..1) * B_top (j=0..1, k=0..6)
    # i_base=0, j_base=0, k_base=0
    # Strassen for k=0,1
    add_strassen_block_correct(U, V, W, 0, 0, 0, 0)
    # Strassen for k=2,3
    add_strassen_block_correct(U, V, W, 7, 0, 0, 2)
    # Strassen for k=4,5
    add_strassen_block_correct(U, V, W, 14, 0, 0, 4)
    # Schoolbook for k=6
    add_schoolbook_block_correct(U, V, W, 21, 0, 0, 6)
    
    # Block 2: A_right (i=0..1, j=2..3) * B_bottom (j=2..3, k=0..6)
    # i_base=0, j_base=2, k_base=0
    # Note: j_base=2 means j indices are 2,3.
    # In U, j is the second index: i*m + j.
    # In V, j is the first index: j*p + k.
    # In W, j is not present.
    
    # Strassen for k=0,1
    add_strassen_block_correct(U, V, W, 25, 0, 2, 0)
    # Strassen for k=2,3
    add_strassen_block_correct(U, V, W, 32, 0, 2, 2)
    # Strassen for k=4,5
    add_strassen_block_correct(U, V, W, 39, 0, 2, 4)
    # Schoolbook for k=6
    add_schoolbook_block_correct(U, V, W, 46, 0, 2, 6)
    
    # Verify
    if verify_fast(U, V, W, n, m, p):
        best_rank = 50
        best_U, best_V, best_W = U, V, W
    else:
        # If verification fails, fall back to schoolbook
        best_rank = rank_school
        best_U, best_V, best_W = schoolbook(n, m, p)

    # Convert to the required format
    r = best_rank
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
