"""Matrix multiplication 2x3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    size_a = n * m  # 6
    size_b = m * p  # 21
    size_c = p * n  # 14
    
    # We will construct a decomposition of rank R = 7 + 14 = 21.
    # Split the inner dimension m=3 into m1=2 and m2=1.
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x7) = [B1 (2x7);  B2 (1x7)]
    # C = A1 @ B1 + A2 @ B2
    
    # Part 1: A1 (2x2) @ B1 (2x7)
    # We can use the 7-term Strassen decomposition for 2x2*2x2, but B1 has 7 columns.
    # The tensor for 2x2*2x7 is a "slice" of the 2x2*2xK problem.
    # Actually, Strassen works for any p if we treat the columns independently?
    # No, Strassen reduces the number of multiplications.
    # For 2x2 * 2x7, the schoolbook rank is 2*2*7 = 28.
    # Strassen reduces the 2x2*2x2 part to 7 terms.
    # Since the 7 columns of B1 are independent, we can apply the same 7 "linear combinations"
    # of A1 entries to each column of B1? 
    # Let's check: C[i,k] = sum_j A1[i,j] B1[j,k].
    # Strassen defines 7 products M_r = (linear combo of A1) * (linear combo of B1 columns?).
    # Actually, Strassen's algorithm computes 7 scalar products for a 2x2*2x2.
    # For 2x2*2x7, we can view it as 7 separate 2x2*2x1 multiplications? No, that's not right.
    # We can view it as a block multiplication.
    # The rank of M_{2,2,7} is known to be 7 * ceil(7/2)? No.
    # Actually, the rank of M_{n,m,p} is at most n*m*p.
    # For M_{2,2,7}, we can use the 7 Strassen terms for the first 2 columns, 
    # and 4 schoolbook terms for the next 2, and 4 for the next 2, and 2 for the last 1?
    # That would be 7+7+7+2 = 23? No, Strassen applies to 2x2 blocks.
    # 7 columns = 2 + 2 + 2 + 1.
    # Block 1 (cols 0,1): 7 terms.
    # Block 2 (cols 2,3): 7 terms.
    # Block 3 (cols 4,5): 7 terms.
    # Block 4 (col 6): 2 terms (schoolbook for 2x2*2x1 is 4, but wait).
    # For 2x2 * 2x1, schoolbook is 4 terms. Can we do better? 
    # M_{2,2,1} rank is 4.
    # So total for A1 @ B1 would be 7 + 7 + 7 + 4 = 25 terms.
    # This is worse than the 14 terms from the schoolbook for the whole 2x3x7 part?
    # Wait, the total schoolbook for 2x3x7 is 42.
    # My previous calculation:
    # A1 @ B1 (2x2 * 2x7): schoolbook 28.
    # A2 @ B2 (2x1 * 1x7): schoolbook 14.
    # Total 42.
    
    # If I use Strassen for 2x2*2x2 blocks:
    # 2x2 * 2x7 can be split into three 2x2*2x2 blocks (using 6 cols) and one 2x2*2x1 block.
    # 3 * 7 + 4 = 25 terms.
    # Plus A2 @ B2 (2x1 * 1x7) = 14 terms.
    # Total 25 + 14 = 39. This is better than 42, but not great.
    
    # Is there a better way for 2x2 * 2x7?
    # The rank of M_{2,2,7} is actually 7 * 2 = 14? No.
    # Let's stick to a simpler constructive approach that is guaranteed to work and be < 42.
    # Actually, the most robust "improvement" over schoolbook (42) is to use the 
    # known rank-7 Strassen for the 2x2 sub-blocks.
    
    # Let's try to construct the 2x2 * 2x7 part with rank 14?
    # If we can prove M_{2,2,7} has rank 14, then total rank is 14 + 14 = 28.
    # But I am not sure about the rank of M_{2,2,7}.
    
    # Let's use the 39-term construction (3x Strassen + 1x Schoolbook + 14 Schoolbook).
    # This is definitely < 42.
    
    R_total = 39
    U = np.zeros((size_a, R_total), dtype=float)
    V = np.zeros((size_b, R_total), dtype=float)
    W = np.zeros((size_c, R_total), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Part 1: A1 (2x2) @ B1 (2x7)
    # Global indices:
    # A1 rows: i=0,1; cols: j=0,1. Global A index: i*3 + j.
    # B1 rows: j=0,1; cols: k=0..6. Global B index: j*7 + k.
    # C rows: i=0,1; cols: k=0..6. Global C index: k*2 + i.
    
    # Strassen terms for 2x2*2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # We apply this to columns (0,1), (2,3), (4,5) of B1.
    # For a pair of columns (k1, k2):
    # b00 -> B[0, k1], b10 -> B[1, k1]
    # b01 -> B[0, k2], b11 -> B[1, k2]
    
    # A indices for 2x2 block (j=0,1):
    # a00 -> A[0, 0] (global 0)
    # a10 -> A[1, 0] (global 3)
    # a01 -> A[0, 1] (global 1)
    # a11 -> A[1, 1] (global 4)
    
    # C indices for outputs corresponding to k1, k2:
    # C[i, k1] -> global k1*2 + i
    # C[i, k2] -> global k2*2 + i
    
    def add_strassen_block(k1, k2):
        nonlocal col
        # M1
        # U: a00+a11 -> U[0]+=1, U[4]+=1
        # V: b00+b11 -> V[k1]+=1, V[7+k2]+=1
        # W: c00+c11 -> W[k1*2+0]+=1, W[k2*2+1]+=1
        U[0, col] = 1; U[4, col] = 1
        V[k1, col] = 1; V[7+k2, col] = 1
        W[k1*2+0, col] = 1; W[k2*2+1, col] = 1
        col += 1
        
        # M2
        # U: a10+a11 -> U[3]+=1, U[4]+=1
        # V: b00 -> V[k1]+=1
        # W: c10+c11? No, M2 contributes to c10 and c11?
        # Strassen output:
        # c00 = M1 - M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # So M2 contributes to c10 and c11 with coeff 1 and -1.
        # W[k1*2+1, col] = 1 (c10)
        # W[k2*2+1, col] = -1 (c11)
        U[3, col] = 1; U[4, col] = 1
        V[k1, col] = 1
        W[k1*2+1, col] = 1; W[k2*2+1, col] = -1
        col += 1
        
        # M3
        # U: a00 -> U[0]+=1
        # V: b10-b11 -> V[7+k1]+=1, V[7+k2]+=-1
        # W: contributes to c01 (coeff 1) and c11 (coeff 1)
        # c01 = M3 + M5 -> W[k2*2+0, col] = 1
        # c11 = M1 - M2 + M3 + M6 -> W[k2*2+1, col] = 1
        U[0, col] = 1
        V[7+k1, col] = 1; V[7+k2, col] = -1
        W[k2*2+0, col] = 1; W[k2*2+1, col] = 1
        col += 1
        
        # M4
        # U: a11 -> U[4]+=1
        # V: b01-b10 -> V[7+k2]+=1, V[7+k1]+=-1
        # W: contributes to c00 (coeff -1) and c10 (coeff 1)
        # c00 = M1 - M4 - M5 + M7 -> W[k1*2+0, col] = -1
        # c10 = M2 + M4 -> W[k1*2+1, col] = 1
        U[4, col] = 1
        V[7+k2, col] = 1; V[7+k1, col] = -1
        W[k1*2+0, col] = -1; W[k1*2+1, col] = 1
        col += 1
        
        # M5
        # U: a00+a01 -> U[0]+=1, U[1]+=1
        # V: b11 -> V[7+k2]+=1
        # W: contributes to c00 (coeff -1) and c01 (coeff 1)
        # c00 = M1 - M4 - M5 + M7 -> W[k1*2+0, col] = -1
        # c01 = M3 + M5 -> W[k2*2+0, col] = 1
        U[0, col] = 1; U[1, col] = 1
        V[7+k2, col] = 1
        W[k1*2+0, col] = -1; W[k2*2+0, col] = 1
        col += 1
        
        # M6
        # U: a10-a00 -> U[3]+=1, U[0]+=-1
        # V: b00+b10 -> V[k1]+=1, V[7+k1]+=1
        # W: contributes to c11 (coeff 1)
        # c11 = M1 - M2 + M3 + M6 -> W[k2*2+1, col] = 1
        U[3, col] = 1; U[0, col] = -1
        V[k1, col] = 1; V[7+k1, col] = 1
        W[k2*2+1, col] = 1
        col += 1
        
        # M7
        # U: a01-a11 -> U[1]+=1, U[4]+=-1
        # V: b01+b11 -> V[7+k2]+=1, V[7+k2]+=1? No, b01 is V[7+k2]? 
        # Wait, b01 is B[0, k2] -> global index 0*7+k2 = k2.
        # b11 is B[1, k2] -> global index 1*7+k2 = 7+k2.
        # So V[k2]+=1, V[7+k2]+=1.
        # W: contributes to c00 (coeff 1)
        # c00 = M1 - M4 - M5 + M7 -> W[k1*2+0, col] = 1
        U[1, col] = 1; U[4, col] = -1
        V[k2, col] = 1; V[7+k2, col] = 1
        W[k1*2+0, col] = 1
        col += 1

    # Apply Strassen to column pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    
    # Remaining column k=6 for A1 @ B1
    # Schoolbook for 2x2 * 2x1
    # C[i, 6] = A1[i,0]B1[0,6] + A1[i,1]B1[1,6]
    # Term 1: A1[0,0] * B1[0,6] -> C[0,6]
    U[0, col] = 1; V[6, col] = 1; W[6*2+0, col] = 1; col += 1
    # Term 2: A1[1,0] * B1[0,6] -> C[1,6]
    U[3, col] = 1; V[6, col] = 1; W[6*2+1, col] = 1; col += 1
    # Term 3: A1[0,1] * B1[1,6] -> C[0,6]
    U[1, col] = 1; V[7+6, col] = 1; W[6*2+0, col] = 1; col += 1
    # Term 4: A1[1,1] * B1[1,6] -> C[1,6]
    U[4, col] = 1; V[7+6, col] = 1; W[6*2+1, col] = 1; col += 1
    
    # Part 2: A2 (2x1) @ B2 (1x7)
    # A2 rows: i=0,1; col: j=2. Global A index: i*3 + 2.
    # B2 row: j=2; cols: k=0..6. Global B index: 2*7 + k = 14 + k.
    # C rows: i=0,1; cols: k=0..6. Global C index: k*2 + i.
    # C[i, k] = A2[i, 2] * B2[2, k]
    
    for i in range(2):
        for k in range(7):
            a_idx = i * 3 + 2
            b_idx = 14 + k
            c_idx = k * 2 + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
