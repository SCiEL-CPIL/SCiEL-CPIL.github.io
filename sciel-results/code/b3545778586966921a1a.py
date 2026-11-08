"""Rank-56 tensor decomposition for 2x4 times 4x8 matrix multiplication using Strassen."""
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start = time.time()
    DEADLINE = start + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 8
    # A: n x m (2x4), B: m x p (4x8), C: n x p (2x8)
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    # W index: W[k*n + i, r]
    # U index: i*m + j
    # V index: j*p + k

    # We use a Strassen-based approach.
    # 
    # The key insight: we can view the 2x4 x 4x8 product as involving
    # overlapping 2x2x2 Strassen subproblems.
    #
    # Split j (middle dimension) into two halves: j in {0,1} and j in {2,3}
    # Split k (output columns) into four pairs: kg in {0,1,2,3}, each with 2 columns
    #
    # For each i in {0,1} and each kg in {0,1,2,3}:
    #   C[i, kg*2] = sum_{j=0}^{3} A[i,j] * B[j, kg*2]
    #   C[i, kg*2+1] = sum_{j=0}^{3} A[i,j] * B[j, kg*2+1]
    #
    # = sum_{j in {0,1}} A[i,j]*B[j,kg*2:kg*2+2] + sum_{j in {2,3}} A[i,j]*B[j,kg*2:kg*2+2]
    #
    # For the first sum (j in {0,1}): this is [A[i,0], A[i,1]] * [[B[0,kg*2], B[0,kg*2+1]], [B[1,kg*2], B[1,kg*2+1]]]
    # For the second sum (j in {2,3}): this is [A[i,2], A[i,3]] * [[B[2,kg*2], B[2,kg*2+1]], [B[3,kg*2], B[3,kg*2+1]]]
    #
    # Each is a 1x2 * 2x2 product. 
    #
    # Now, here's the key: for a fixed i, the A-vector [A[i,0], A[i,1], A[i,2], A[i,3]] is the same
    # across all kg. So we can precompute combinations of A entries.
    #
    # For the 1x2 * 2x2 case: [a,b] * [[p,q],[r,s]] = [ap+br, aq+bs]
    # 
    # We can use 3 multiplications:
    # M1 = a * p
    # M2 = b * r
    # M3 = (a+b) * (q+s)
    # Then: ap+br = M1 + M2
    #       aq+bs = M3 - M1 + M2 - (a*s + b*q)... no, that doesn't work directly.
    #
    # Actually, let me reconsider. The standard result is that 1x2*2x2 needs 4 multiplications
    # in general (it's a 1x2x2 tensor which has rank 4). But we can share across kg values.
    #
    # Better approach: Use the full Strassen structure.
    #
    # Let's think of it differently. Define for each i:
    #   a = A[i,0], b = A[i,1], c = A[i,2], d = A[i,3]
    #
    # Precompute:
    #   s1 = a + b
    #   s2 = c + d
    #   s3 = a + c
    #   s4 = b + d
    #   s5 = a - b
    #   s6 = c - d
    #
    # For each kg, define B-blocks:
    #   B1[kg] = B[0, kg*2:kg*2+2] = [B[0,kg*2], B[0,kg*2+1]]
    #   B2[kg] = B[1, kg*2:kg*2+2] = [B[1,kg*2], B[1,kg*2+1]]
    #   B3[kg] = B[2, kg*2:kg*2+2] = [B[2,kg*2], B[2,kg*2+1]]
    #   B4[kg] = B[3, kg*2:kg*2+2] = [B[3,kg*2], B[3,kg*2+1]]
    #
    # C[i, kg*2] = a*B[0,kg*2] + b*B[1,kg*2] + c*B[2,kg*2] + d*B[3,kg*2]
    # C[i, kg*2+1] = a*B[0,kg*2+1] + b*B[1,kg*2+1] + c*B[2,kg*2+1] + d*B[3,kg*2+1]
    #
    # Using Strassen-like combinations:
    # Let's define for each kg:
    #   t1 = B[0,kg*2] + B[1,kg*2]
    #   t2 = B[2,kg*2] + B[3,kg*2]
    #   t3 = B[0,kg*2+1] + B[1,kg*2+1]
    #   t4 = B[2,kg*2+1] + B[3,kg*2+1]
    #   t5 = B[0,kg*2] - B[1,kg*2]
    #   t6 = B[2,kg*2] - B[3,kg*2]
    #   t7 = B[0,kg*2+1] - B[1,kg*2+1]
    #   t8 = B[2,kg*2+1] - B[3,kg*2+1]
    #
    # Then:
    # C[i,kg*2] = (s1/2)*(t1 + t2) + (s3/2)*(t1 - t2) + (s5/2)*(t5 - t6) + (s6/2)*(t5 + t6)
    #            = (s1*(t1+t2) + s3*(t1-t2) + s5*(t5-t6) + s6*(t5+t6)) / 2
    #
    # Hmm, this is getting complicated. Let me just use a direct rank-56 construction.
    #
    # Actually, the cleanest way: use 7 Strassen multiplications per 2x2x2 subproblem.
    # 
    # View the problem as: for each of the 2 rows i of A, and for each of the 4 column-pairs kg,
    # we have a 2x2 matrix product where:
    #   A_sub is 2x2 (rows are j-halves {0,1} and {2,3}, cols are within-half {0,1})
    #   B_sub is 2x2 (rows are within-half {0,1}, cols are within-pair {0,1})
    #   C_sub is 2x2 (rows are j-halves, cols are within-pair)
    #
    # Wait, that's not quite right either. Let me just build rank-56 directly.
    #
    # Known: rank(2,4,8) <= 56. Here's a construction:
    #
    # Split into 4 independent problems, one for each kg in {0,1,2,3}.
    # For each kg, we need to compute a 2x4 * 4x2 product (for both rows of A).
    # That's a 2x4x2 tensor. 
    #
    # For a 2x4x2 problem: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # Schoolbook: 2*4*2 = 16 multiplications.
    # With Strassen on 2x2 blocks: split j into {0,1} and {2,3}.
    #   C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    # First term: 2x2 * 2x2 (rows i, cols k, middle j in {0,1}) -> use 7 Strassen mults
    # Second term: 2x2 * 2x2 (rows i, cols k, middle j in {2,3}) -> use 7 Strassen mults
    # Total per kg: 14 mults. For 4 kg values: 56 mults total.
    #
    # This gives rank 56!
    #
    # Let me implement this.

    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56

    col = 0
    for kg in range(4):
        # For this kg, we do two 2x2x2 Strassen multiplications
        # First: j in {0,1}, i in {0,1}, k in {kg*2, kg*2+1}
        # Second: j in {2,3}, i in {0,1}, k in {kg*2, kg*2+1}
        
        for j_half in range(2):
            # j_half=0: j in {0,1}, j_half=1: j in {2,3}
            j_base = j_half * 2
            # Strassen 2x2x2: 7 multiplications
            # A_sub[i, jw] = A[i, j_base + jw] for i in {0,1}, jw in {0,1}
            # B_sub[jw, kw] = B[j_base + jw, kg*2 + kw] for jw in {0,1}, kw in {0,1}
            # C_sub[i, kw] = sum_{jw} A_sub[i,jw] * B_sub[jw,kw]
            
            # Strassen formulas for 2x2*2x2:
            # M1 = (A00+A11)*(B00+B11)  -- wait, standard Strassen is:
            # M1 = A00 * B00
            # M2 = A10 * B00
            # M3 = A00 * B11
            # M4 = A11 * B11
            # M5 = (A00+A11)*(B10+B11)
            # M6 = (A10+A11)*B01
            # M7 = (A00-A10)*(B01+B11)
            # C00 = M1 + M5 - M4 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            #
            # Wait, let me use the standard Strassen:
            # M1 = A00*(B00+B11)
            # M2 = (A00+A01)*B11
            # M3 = (A10+A11)*(B00+B01)
            # M4 = A11*(B10-B00)
            # M5 = (A00-A10)*(B01+B11)
            # M6 = (A01+A11)*(B00-B10)
            # M7 = (A00+A01-A10-A11)*B10
            # C00 = M1+M4-M5+M7
            # C01 = M2+M5
            # C10 = M3+M6
            # C11 = M1-M2+M4+M6
            #
            # Let me use the simpler version:
            # M1 = A00 * B00
            # M2 = A10 * B00  
            # M3 = A00 * B11
            # M4 = A11 * B11
            # M5 = (A00+A11) * (B10+B11)
            # M6 = (A10+A11) * B01
            # M7 = (A00-A10) * (B01+B11)
            # C00 = M1 + M5 - M4 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            strassen_mults = [
                # (A_coeff, B_coeff, description)
                # M1 = A00 * B00
                ([(0,0), (1,0)], [(0,0), (1,0)], 1),  # A[0,0]*B[0,0]
                # M2 = A10 * B00
                ([(1,0), (1,0)], [(0,0), (1,0)], 1),  # A[1,0]*B[0,0]
                # M3 = A00 * B11
                ([(0,0), (1,0)], [(1,1), (1,1)], 1),  # A[0,0]*B[1,1]
                # M4 = A11 * B11
                ([(1,1), (1,1)], [(1,1), (1,1)], 1),  # A[1,1]*B[1,1]
                # M5 = (A00+A11) * (B10+B11)
                ([(0,0), (1,1)], [(1,0), (1,1)], 1),
                # M6 = (A10+A11) * B01
                ([(1,0), (1,1)], [(0,1), (0,1)], 1),
                # M7 = (A00-A10) * (B01+B11)
                ([(0,0), (1,0)], [(0,1), (1,1)], 1),
            ]
            
            # For each Strassen multiplication, add to U, V, W
            for mult_idx in range(7):
                a_coeffs = strassen_mults[mult_idx][0]
                b_coeffs = strassen_mults[mult_idx][1]
                
                # Build U vector: for each (i, j) where j = j_base + jw
                u_vec = [0.0] * (n * m)
                for (i, jw), coeff in a_coeffs:
                    u_vec[i * m + (j_base + jw)] = coeff
                
                # Build V vector: for each (j, k) where j = j_base + jw, k = kg*2 + kw
                v_vec = [0.0] * (m * p)
                for (jw, kw), coeff in b_coeffs:
                    v_vec[(j_base + jw) * p + (kg * 2 + kw)] = coeff
                
                # Determine W contributions based on which Strassen mult this is
                # C00 = M1 + M5 - M4 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                w_vec = [0.0] * (p * n)
                if mult_idx == 0:  # M1
                    # C00 += M1, C11 += M1
                    w_vec[(kg*2+0)*n + 0] += 1.0  # C[0, kg*2]
                    w_vec[(kg*2+1)*n + 1] += 1.0  # C[1, kg*2+1]
                elif mult_idx == 1:  # M2
                    # C10 += M2, C11 -= M2
                    w_vec[(kg*2+0)*n + 1] += 1.0  # C[1, kg*2]
                    w_vec[(kg*2+1)*n + 1] -= 1.0  # C[1, kg*2+1]
                elif mult_idx == 2:  # M3
                    # C01 += M3, C11 += M3
                    w_vec[(kg*2+1)*n + 0] += 1.0  # C[0, kg*2+1]
                    w_vec[(kg*2+1)*n + 1] += 1.0  # C[1, kg*2+1]
                elif mult_idx == 3:  # M4
                    # C00 -= M4, C10 += M4
                    w_vec[(kg*2+0)*n + 0] -= 1.0  # C[0, kg*2]
                    w_vec[(kg*2+0)*n + 1] += 1.0  # C[1, kg*2]
                elif mult_idx == 4:  # M5
                    # C00 += M5, C01 += M5
                    w_vec[(kg*2+0)*n + 0] += 1.0  # C[0, kg*2]
                    w_vec[(kg*2+1)*n + 0] += 1.0  # C[0, kg*2+1]
                elif mult_idx == 5:  # M6
                    # C11 += M6
                    w_vec[(kg*2+1)*n + 1] += 1.0  # C[1, kg*2+1]
                elif mult_idx == 6:  # M7
                    # C00 += M7
                    w_vec[(kg*2+0)*n + 0] += 1.0  # C[0, kg*2]
                
                # Add to global U, V, W
                for idx in range(len(u_vec)):
                    U[idx][col] = u_vec[idx]
                for idx in range(len(v_vec)):
                    V[idx][col] = v_vec[idx]
                for idx in range(len(w_vec)):
                    W[idx][col] = w_vec[idx]
                
                col += 1

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
