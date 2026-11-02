import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We want to decompose the matrix multiplication tensor T_{i,j,k} = e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # Strategy:
    # 1. Split the multiplication into blocks.
    # 2. Use Strassen's algorithm for 2x2x2 blocks where possible.
    # 3. Use schoolbook for the remainder.
    
    # A is 4x4, B is 4x7, C is 4x7.
    # Let's split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2
    
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4)
    # Part 2: A (4x4) * B2 (4x3) -> C2 (4x3)
    
    # We can further split Part 1 and Part 2 into 2x2 blocks.
    # Let A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B1 = [[B11, B12], [B21, B22]] where each block is 2x2.
    # B2 = [[B13, B14, B15], [B23, B24, B25]] where each block is 2x3.
    
    # Actually, let's just use the general Strassen decomposition for 2x2x2.
    # The rank of 2x2x2 is 7.
    # The rank of 4x4x4 is at most 7 * 7 = 49 (if we apply Strassen to each of the 7 blocks of the 2x2x2 block structure? No, that's not right).
    # Actually, if we view 4x4x4 as 2x2 blocks of 2x2 matrices, the number of multiplications is 7 * (rank of 2x2x2) = 7 * 7 = 49?
    # No, the 7 terms in Strassen for 2x2x2 involve products of 2x2 matrices.
    # If we apply this to the block structure, we get 7 multiplications of 2x2 blocks.
    # Each 2x2 block multiplication has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # For 4x4x3, we can split B2 into B2a (4x2) and B2b (4x1).
    # Rank 4x4x2: Split into 2x2 blocks. 2x2x2 blocks.
    # A (4x4) is 2x2 blocks of 2x2. B2a (4x2) is 2x1 blocks of 2x2.
    # This is getting complicated.
    
    # Let's stick to a simpler, verified improvement:
    # Use Strassen for the 4x4x4 part (rank 49) and schoolbook for 4x4x3 (rank 48).
    # Total rank: 97.
    
    # But wait, can we do better for 4x4x3?
    # 4x4x3 can be split into 4x4x2 + 4x4x1.
    # 4x4x2: Split into 2x2x2 blocks? No, dimensions don't match nicely for Strassen.
    # Let's just use Strassen for the 4x4x4 part.
    
    # Implementation of Strassen 2x2x2 terms:
    # The 7 rank-1 terms (u, v, w) for 2x2x2 are:
    # 1. (1,0,0,1) ⊗ (1,0,0,1) ⊗ (1,0,0,1)
    # 2. (0,0,1,1) ⊗ (1,0,0,0) ⊗ (0,0,1,1)
    # 3. (1,0,0,0) ⊗ (0,1,0,-1) ⊗ (0,1,0,-1)
    # 4. (0,0,0,1) ⊗ (0,0,1,-1) ⊗ (0,0,1,-1)
    # 5. (1,1,0,0) ⊗ (0,0,0,1) ⊗ (1,1,0,0)
    # 6. (0,1,-1,0) ⊗ (1,1,0,0) ⊗ (0,1,1,-1)
    # 7. (0,1,0,-1) ⊗ (0,0,1,1) ⊗ (0,1,-1,1)
    
    # Let's verify these terms sum to the 2x2x2 multiplication tensor.
    # T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    strassen_2x2_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),
        ([0,0,0,1], [0,0,1,-1], [0,0,1,-1]),
        ([1,1,0,0], [0,0,0,1], [1,1,0,0]),
        ([0,1,-1,0], [1,1,0,0], [0,1,1,-1]),
        ([0,1,0,-1], [0,0,1,1], [0,1,-1,1])
    ]
    
    # Check if these are correct.
    # Let's do a quick check for T[0,0,0].
    # T[0,0,0] should be 1.
    # Term 1: u[0,0]=1, v[0,0]=1, w[0,0]=1 -> 1
    # Term 2: u[0,0]=0 -> 0
    # Term 3: u[0,0]=1, v[0,0]=0 -> 0
    # Term 4: u[0,0]=0 -> 0
    # Term 5: u[0,0]=1, v[0,0]=0 -> 0
    # Term 6: u[0,0]=0 -> 0
    # Term 7: u[0,0]=0 -> 0
    # Sum = 1. Correct.
    
    # Check T[0,0,1]. Should be 0.
    # Term 1: u[0,0]=1, v[0,1]=0 -> 0
    # Term 2: u[0,0]=0 -> 0
    # Term 3: u[0,0]=1, v[0,1]=1, w[1,0]=0 -> 0
    # Term 4: u[0,0]=0 -> 0
    # Term 5: u[0,0]=1, v[0,1]=0 -> 0
    # Term 6: u[0,0]=0 -> 0
    # Term 7: u[0,0]=0 -> 0
    # Sum = 0. Correct.
    
    # Check T[0,1,0]. Should be 0.
    # Term 1: u[0,1]=0 -> 0
    # Term 2: u[0,1]=0 -> 0
    # Term 3: u[0,1]=0 -> 0
    # Term 4: u[0,1]=0 -> 0
    # Term 5: u[0,1]=1, v[1,0]=0 -> 0
    # Term 6: u[0,1]=1, v[1,0]=0 -> 0
    # Term 7: u[0,1]=1, v[1,0]=0 -> 0
    # Sum = 0. Correct.
    
    # Check T[0,1,1]. Should be 1.
    # Term 1: u[0,1]=0 -> 0
    # Term 2: u[0,1]=0 -> 0
    # Term 3: u[0,1]=0 -> 0
    # Term 4: u[0,1]=0 -> 0
    # Term 5: u[0,1]=1, v[1,1]=1, w[1,0]=0 -> 0. Wait.
    # w[1,0] is index 2 in w? w is [w00, w01, w10, w11].
    # w[1,1] is index 3.
    # Term 5: w = [1,1,0,0]. w[1,1] = 0.
    # Let's re-evaluate Term 5.
    # u=[1,1,0,0], v=[0,0,0,1], w=[1,1,0,0].
    # T[0,1,1] += u[0,1]*v[1,1]*w[1,0] = 1 * 1 * 0 = 0.
    # Term 6: u=[0,1,-1,0], v=[1,1,0,0], w=[0,1,1,-1].
    # T[0,1,1] += u[0,1]*v[1,1]*w[1,0] = 1 * 0 * 1 = 0.
    # Term 7: u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,-1,1].
    # T[0,1,1] += u[0,1]*v[1,1]*w[1,0] = 1 * 1 * (-1) = -1.
    # Sum so far: 0 + 0 + 0 + 0 + 0 + 0 + (-1) = -1.
    # But T[0,1,1] should be 1.
    # My Strassen terms are likely incorrect or I am summing them wrong.
    
    # Given the risk and time, I will use a known correct Strassen decomposition.
    # Reference: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # This corresponds to the following rank-1 terms in the tensor decomposition:
    # Each M_i is a rank-1 term in the "multiplication" sense, but we need to express the final C as a sum of rank-1 tensors.
    # Actually, the tensor decomposition of rank 7 is exactly the set of these 7 terms.
    # The vectors u, v, w for each term are:
    # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (This is M1? No, M1 contributes to C00, C11 with sign +, and to others with sign - or +).
    
    # Let's construct the terms carefully.
    # Term 1 (M1): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    #   Contribution to C00: +1, C11: +1.
    # Term 4 (M4): u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,-1]
    #   Contribution to C00: -1, C10: +1, C11: -1.
    #   Wait, M4 = A11(B10-B00).
    #   C10 += M4, C11 -= M4.
    #   So w for M4 should be [0,0,1,-1] (C10 +=, C11 -=).
    # Term 6 (M6): u=[0,1,-1,0], v=[1,1,0,0], w=[0,1,1,-1]
    #   M6 = (A10-A00)(B00+B01).
    #   C00 += M6, C01 += M6, C10 += M6, C11 -= M6.
    #   So w should be [1,1,1,-1].
    # Term 7 (M7): u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,-1,1]
    #   M7 = (A01-A11)(B10+B11).
    #   C00 -= M7, C01 += M7, C10 -= M7, C11 += M7.
    #   So w should be [-1,1,-1,1].
    
    # Let's list all 7 terms with correct w vectors:
    # 1. M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # 2. M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]  (C10 += M2)
    # 3. M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,-1] (C01 += M3, C11 -= M3)
    # 4. M4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,-1] (C10 += M4, C11 -= M4, C00 -= M4) -> w=[-1,0,1,-1]
    # 5. M5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]  (C01 += M5, C11 += M5, C00 += M5) -> w=[1,1,0,0]
    # 6. M6: u=[0,1,-1,0], v=[1,1,0,0], w=[1,1,1,-1]
    # 7. M7: u=[0,1,0,-1], v=[0,0,1,1], w=[-1,1,-1,1]
    
    # Let's verify C00 = M1 - M4 + M6 - M7.
    # M1: C00 += 1
    # M4: C00 -= 1 (from w[0,0]=-1)
    # M6: C00 += 1 (from w[0,0]=1)
    # M7: C00 -= 1 (from w[0,0]=-1)
    # Sum: 1 - 1 + 1 - 1 = 0.
    # But C00 should be A00B00 + A01B10.
    # My w vectors are not capturing the full contribution.
    
    # Given the complexity and risk of error, I will use the schoolbook decomposition for the entire 4x4x7 problem.
    # Rank = 4*4*7 = 112.
    # This is the baseline.
    
    # To improve, I will use Strassen for the 4x4x4 part (rank 49) and schoolbook for 4x4x3 (rank 48).
    # Total rank 97.
    
    # I will implement the Strassen decomposition for 4x4x4 by applying the 2x2x2 Strassen algorithm to the 2x2 block structure.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Strassen 2x2x2 terms (u, v, w) are in R^4.
    # We need to map these to the 4x4x4 block structure.
    # Let's define the 7 terms for 2x2x2.
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),
        ([0,0,0,1], [0,0,1,-1], [-1,0,1,-1]),
        ([1,1,0,0], [0,0,0,1], [1,1,0,0]),
        ([0,1,-1,0], [1,1,0,0], [1,1,1,-1]),
        ([0,1,0,-1], [0,0,1,1], [-1,1,-1,1])
    ]
    
    # For 4x4x4, we have 2x2 blocks of 2x2 matrices.
    # Let A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]].
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # We can apply Strassen to each of the 4 block multiplications? No, that would be 4 * 7 = 28.
    # But we can apply Strassen to the whole 2x2 block structure.
    # The 7 Strassen terms for the 2x2 block multiplication are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # Each M_i is a product of 2x2 matrices.
    # We can decompose each M_i into rank-1 terms using the 2x2x2 Strassen algorithm.
    # But that would be 7 * 7 = 49 terms.
    
    # Let's implement this.
    
    def add_strassen_2x2_block(u_block, v_block, w_block, i_off, j_off, k_off):
        # u_block, v_block, w_block are 2x2 matrices (lists of lists).
        # We need to add 7 rank-1 terms for this 2x2x2 multiplication.
        for (u, v, w) in strassen_terms:
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            
            # u is [u00, u01, u10, u11]
            u00, u01, u10, u11 = u
            v00, v01, v10, v11 = v
            w00, w01, w10, w11 = w
            
            # Map to global indices
            # i = i_off + 0 or 1
            # j = j_off + 0 or 1
            # k = k_off + 0 or 1
            
            u_vec[(i_off + 0)*m + (j_off + 0)] = u00
            u_vec[(i_off + 0)*m + (j_off + 1)] = u01
            u_vec[(i_off + 1)*m + (j_off + 0)] = u10
            u_vec[(i_off + 1)*m + (j_off + 1)] = u11
            
            v_vec[(j_off + 0)*p + (k_off + 0)] = v00
            v_vec[(j_off + 0)*p + (k_off + 1)] = v01
            v_vec[(j_off + 1)*p + (k_off + 0)] = v10
            v_vec[(j_off + 1)*p + (k_off + 1)] = v11
            
            w_vec[(k_off + 0)*n + (i_off + 0)] = w00
            w_vec[(k_off + 0)*n + (i_off + 1)] = w01
            w_vec[(k_off + 1)*n + (i_off + 0)] = w10
            w_vec[(k_off + 1)*n + (i_off + 1)] = w11
            
            add_term(u_vec, v_vec, w_vec)
    
    # Now, we need to define the 7 block multiplications M1..M7.
    # Each M_i is a product of two 2x2 matrices.
    # We can represent each M_i as a sum of rank-1 terms using the 2x2x2 Strassen algorithm.
    # But we need to be careful: the 7 Strassen terms for 2x2x2 are for the multiplication of two 2x2 matrices.
    # So for each M_i, we can use the 7 terms.
    
    # Let's define the 7 block products.
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We can't just add the Strassen terms for each M_i, because the M_i are not independent.
    # The correct way is to use the 7 Strassen terms for the 2x2 block structure, and then decompose each resulting 2x2 product into rank-1 terms.
    
    # This is getting too complex. I will fall back to schoolbook for the entire problem.
    
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
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
