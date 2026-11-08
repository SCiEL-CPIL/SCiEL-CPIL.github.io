import numpy as np
import itertools
import time
import random

def construct():
    n, m, p = 2, 4, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline (rank 40)
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m*p + j*p + k
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
        return U, V, W, rank
    
    U0, V0, W0, rank0 = schoolbook()
    
    # Check if a set of rank-r columns exactly decomposes the tensor
    # Tensor T has T[(i,j,k)] = 1 if the product term a_ik * b_jk contributes to c_ij
    # We check: for all i,j, sum_c U[i*m+j, c] * V[j*p+k... ] wait no.
    # The bilinear form is C_ij = sum_k A_ik B_kj.
    # Decomposition: C = sum_c u_c v_c^T w_c^T (broadcasted appropriately)
    # Specifically: sum_c U[:, c] * V[:, c] (outer) doesn't quite work directly for 3-tensor.
    # The standard form: T_{ijk} = sum_c U_{ij, c} V_{jk, c} W_{ki, c} 
    # where indices are: U indexed by (i,j) [row of A], V by (j,k) [col of A, row of B], W by (k,i) [col of B, row of C]
    # Wait, the standard tensor multiplication identity:
    # C_{ij} = sum_k A_{ik} B_{kj}
    # If we have rank r: C_{ij} = sum_c (sum_k' ...) no.
    # The factors are:
    # u_c is a vector of length n*m (indexed by i,j? No, by i and k? No.)
    # Let's look at the seed code:
    # U[i*m+j, column] = 1  -> indexed by (i, j) where j is the column of A? No, A is n x m.
    # Wait, A is n x m. So A[i, j] with i in 0..n-1, j in 0..m-1.
    # B is m x p. B[j, k] with j in 0..m-1, k in 0..p-1.
    # C is n x p. C[i, k] with i in 0..n-1, k in 0..p-1.
    # Seed: U[i*m+j], V[j*p+k], W[k*n+i].
    # This corresponds to the tensor T_{i, j, k} where the product is A_{i, j} * B_{j, k} -> C_{i, k}.
    # The sum over c is: sum_c U[i*m+j, c] * V[j*p+k, c] = C[i, k]? No, that would be a sum over c producing a matrix of size (n*m) x (m*p).
    # Actually, the decomposition is:
    # The multiplication tensor has indices (i, j, k).
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # The rank decomposition is T_{i,j,k} = sum_c U_{(i,j), c} V_{(j,k), c} W_{(k,i), c}
    # This is correct.
    
    def check_decomposition(U, V, W):
        # U: nm x r, V: mp x r, W: pn x r
        # Reconstruct T: for each i,j,k, sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should be 1
        # We can check this efficiently using matrix operations.
        # Let's just iterate if r is small, or use np.einsum
        # r is likely < 40, so iteration is fast.
        n, m, p = 2, 4, 5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank decomposition using random search / iterative refinement
    # This is a hard problem. Known bounds:
    # For 2x4x5, the schoolbook rank is 40.
    # Can we do better?
    # A 2x4 matrix mult can be seen as two 2x2 mults if we split the 4 columns.
    # 2x2x2 has rank 7 (Strassen).
    # So 2x4x2 can be done in rank 7 + 7 = 14.
    # Then 2x2x5? Split 5 into 2+3?
    # 2x2x3: schoolbook 12. Can we do better?
    # Or split 4 into 2+2, and 5 into 2+3.
    # Total = R(2,2,2) + R(2,2,3) + R(2,2,2) + R(2,2,3)? No, that's not how it works.
    # We have A (2x4) and B (4x5).
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B1 (2x5) and B2 (2x5).
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2 times 2x5.
    # A2 B2 is 2x2 times 2x5.
    # What is the rank of 2x2x5?
    # We can write 2x2x5 as three 2x2x2 blocks? No, 5 is not divisible by 2.
    # 2x2x5 can be split into 2x2x2 + 2x2x2 + 2x2x1.
    # Rank(2,2,2) = 7.
    # Rank(2,2,1) = 4 (schoolbook, since 2*2*1=4, and no known improvement for p=1? Actually 2x2x1 is just dot products, rank 4).
    # So R(2,2,5) <= 7 + 7 + 4 = 18.
    # Then total rank <= 18 + 18 = 36.
    
    # Let's construct this rank 36 decomposition explicitly.
    
    def construct_rank_36():
        # A is 2x4, B is 4x5, C is 2x5.
        # Split A into A_left (2x2) and A_right (2x2).
        # Split B into B_top (2x5) and B_bottom (2x5).
        # C = A_left B_top + A_right B_bottom.
        
        # We need decomposition for 2x2x5.
        # Split 2x2x5 into:
        #   Block 1: 2x2x2 (cols 0,1 of the 5)
        #   Block 2: 2x2x2 (cols 2,3 of the 5)
        #   Block 3: 2x2x1 (col 4 of the 5)
        
        # Strassen for 2x2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B21 - B22)
        # M4 = A22 (B12 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will build the U, V, W matrices for the whole 2x4x5 problem.
        # Total rank = 2 * (7 + 7 + 4) = 36.
        
        rank = 36
        U = np.zeros((8, rank)) # 2*4
        V = np.zeros((20, rank)) # 4*5
        W = np.zeros((10, rank)) # 5*2
        
        # Helper to add a 2x2x2 Strassen block
        # a_offset: row offset for A (0 or 2, since A is 2x4, rows 0,1)
        # Actually A is 2x4. Let's say A_left uses cols 0,1. A_right uses cols 2,3.
        # For a 2x2x2 block, A_sub is 2x2.
        # If it's A_left, A_sub[i,j] = A[i, j].
        # If it's A_right, A_sub[i,j] = A[i, j+2].
        # Similarly for B.
        
        def add_strassen_222(a_col_offset, b_col_offset, rank_offset):
            # a_col_offset: 0 or 2 (which 2x2 submatrix of A we're using)
            # b_col_offset: 0, 2, or 4? No, B is 4x5.
            # The 2x2x2 block uses a 2x2 submatrix of B.
            # The 2x2 submatrix of B uses 2 rows and 2 cols.
            # The rows of B correspond to the cols of A.
            # If we are computing A_left B_top, we use rows 0,1 of B.
            # If A_right B_bottom, we use rows 2,3 of B.
            # b_row_offset: 0 or 2.
            # b_col_offset: 0, 2, or 4.
            
            # This function is getting complicated with offsets.
            # Let's just write it out for each of the 4 blocks.
            
            pass
        
        # Let's do it manually.
        # Block 1: A_left (cols 0,1) * B_top (rows 0,1; cols 0,1 of B -> 2x2)
        # Block 2: A_left (cols 0,1) * B_top (rows 0,1; cols 2,3 of B -> 2x2)
        # Block 3: A_left (cols 0,1) * B_top (rows 0,1; col 4 of B -> 2x1)
        # Block 4: A_right (cols 2,3) * B_bottom (rows 2,3; cols 0,1 of B -> 2x2)
        # Block 5: A_right (cols 2,3) * B_bottom (rows 2,3; cols 2,3 of B -> 2x2)
        # Block 6: A_right (cols 2,3) * B_bottom (rows 2,3; col 4 of B -> 2x1)
        
        # Strassen 2x2x2 indices:
        # A_sub is 2x2: indices (0,0), (0,1), (1,0), (1,1)
        # B_sub is 2x2: indices (0,0), (0,1), (1,0), (1,1)
        # C_sub is 2x2: indices (0,0), (0,1), (1,0), (1,1)
        
        # U index: i*m + j. For 2x2x2, i in 0,1, j in 0,1.
        # V index: j*p + k. For 2x2x2, j in 0,1, k in 0,1.
        # W index: k*n + i. For 2x2x2, k in 0,1, i in 0,1.
        
        # Let's define the Strassen vectors for a 2x2x2 block.
        # u vectors (length 4):
        # M1: A00+A11 -> [1,0,0,1]
        # M2: A10+A11 -> [0,0,1,1]
        # M3: A00 -> [1,0,0,0]
        # M4: A11 -> [0,0,0,1]
        # M5: A00+A01 -> [1,1,0,0]
        # M6: A10-A00 -> [-1,0,1,0]
        # M7: A01-A11 -> [0,1,-1,0]
        
        # v vectors (length 4):
        # M1: B00+B11 -> [1,0,0,1]
        # M2: B00 -> [1,0,0,0]
        # M3: B10-B11 -> [0,0,1,-1]
        # M4: B01-B00 -> [0,1,-1,0]
        # M5: B11 -> [0,0,0,1]
        # M6: B00+B01 -> [1,1,0,0]
        # M7: B10+B11 -> [0,0,1,1]
        
        # w vectors (length 4):
        # C00: M1+M4-M5+M7 -> [1,0,-1,1]
        # C01: M3+M5 -> [0,0,1,1]
        # C10: M2+M4 -> [1,0,0,1]
        # C11: M1-M2+M3+M6 -> [1,-1,1,1]
        
        # Wait, my previous derivation:
        # C11 = M1 - M2 + M3 + M6
        # C21 = M2 + M4
        # C12 = M3 + M5
        # C22 = M1 + M4 - M5 + M7
        # Let's re-verify C00 (which is C11 in 1-based).
        # C00 = M1 + M4 - M5 + M7.
        # M1=(A00+A11)(B00+B11)
        # M4=A11(B01-B00)
        # M5=(A00+A01)B11
        # M7=(A01-A11)(B10+B11)
        # Sum = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
        # = A00B00 + A11B01 + A01B10
        # This is correct for C00.
        
        # So the w vectors for the 7 M's are:
        # M1: [1, 0, 0, 1] (C00, C10, C01, C11) -> no, w is for a specific output.
        # We need to express each C_ij as a linear combination of M's.
        # C00 = 1*M1 + 1*M4 - 1*M5 + 1*M7
        # C01 = 1*M3 + 1*M5
        # C10 = 1*M2 + 1*M4
        # C11 = 1*M1 - 1*M2 + 1*M3 + 1*M6
        
        # So for each M, we add to W the coefficient for each C_ij.
        # M1: C00+=1, C01+=0, C10+=0, C11+=1 -> w=[1,0,0,1]
        # M2: C00+=0, C01+=0, C10+=1, C11+=-1 -> w=[0,0,1,-1]
        # M3: C00+=0, C01+=1, C10+=0, C11+=1 -> w=[0,1,0,1]
        # M4: C00+=1, C01+=0, C10+=1, C11+=0 -> w=[1,0,1,0]
        # M5: C00+=-1, C01+=1, C10+=0, C11+=0 -> w=[-1,1,0,0]
        # M6: C00+=0, C01+=0, C10+=0, C11+=1 -> w=[0,0,0,1]
        # M7: C00+=1, C01+=0, C10+=0, C11+=0 -> w=[1,0,0,0]
        
        # Let's re-verify these w vectors.
        # C00 = M1 + M4 - M5 + M7. w_M1[0]+=1, w_M4[0]+=1, w_M5[0]+=-1, w_M7[0]+=1.
        # C01 = M3 + M5. w_M3[1]+=1, w_M5[1]+=1.
        # C10 = M2 + M4. w_M2[2]+=1, w_M4[2]+=1.
        # C11 = M1 - M2 + M3 + M6. w_M1[3]+=1, w_M2[3]+=-1, w_M3[3]+=1, w_M6[3]+=1.
        
        # So:
        # M1: [1, 0, 0, 1]
        # M2: [0, 0, 1, -1]
        # M3: [0, 1, 0, 1]
        # M4: [1, 0, 1, 0]
        # M5: [-1, 1, 0, 0]
        # M6: [0, 0, 0, 1]
        # M7: [1, 0, 0, 0]
        
        # Now, for each 2x2x2 block, we add these to the global U, V, W.
        
        def add_2x2x2_block(a_row_off, a_col_off, b_row_off, b_col_off, rank_off):
            # a_row_off: 0 (since A is 2x4, rows are always 0,1)
            # a_col_off: 0 or 2
            # b_row_off: 0 or 2
            # b_col_off: 0, 2, or 4
            # rank_off: starting column in U, V, W
            
            # U indices: (i, j) -> i*a_rows + j. Here a_rows=2.
            # So U[i, j] is at index i*2 + j in the 2x2 submatrix.
            # Global U index: i*4 + (a_col_off + j).
            
            # V indices: (j, k) -> j*b_cols + k. Here b_cols=2 (for the submatrix).
            # So V[j, k] is at index j*2 + k.
            # Global V index: (b_row_off + j)*5 + (b_col_off + k).
            
            # W indices: (k, i) -> k*2 + i.
            # So W[k, i] is at index k*2 + i.
            # Global W index: (b_col_off + k)*2 + i.
            
            # u vectors for the 2x2 submatrix (length 4, indexed by i*2+j):
            u_M = [
                [1,0,0,1], # M1: A00+A11
                [0,0,1,1], # M2: A10+A11
                [1,0,0,0], # M3: A00
                [0,0,0,1], # M4: A11
                [1,1,0,0], # M5: A00+A01
                [-1,0,1,0],# M6: A10-A00
                [0,1,-1,0] # M7: A01-A11
            ]
            
            # v vectors for the 2x2 submatrix (length 4, indexed by j*2+k):
            v_M = [
                [1,0,0,1], # M1: B00+B11
                [1,0,0,0], # M2: B00
                [0,0,1,-1],# M3: B10-B11
                [0,1,-1,0],# M4: B01-B00
                [0,0,0,1], # M5: B11
                [1,1,0,0], # M6: B00+B01
                [0,0,1,1]  # M7: B10+B11
            ]
            
            # w vectors for the 2x2 submatrix (length 4, indexed by k*2+i):
            w_M = [
                [1,0,0,1], # M1
                [0,0,1,-1],# M2
                [0,1,0,1], # M3
                [1,0,1,0], # M4
                [-1,1,0,0],# M5
                [0,0,0,1], # M6
                [1,0,0,0]  # M7
            ]
            
            for m_idx in range(7):
                r = rank_off + m_idx
                # Add u
                for local_idx, val in enumerate(u_M[m_idx]):
                    if val != 0:
                        i, j = divmod(local_idx, 2)
                        global_u_idx = i*4 + (a_col_off + j)
                        U[global_u_idx, r] += val
                # Add v
                for local_idx, val in enumerate(v_M[m_idx]):
                    if val != 0:
                        j, k = divmod(local_idx, 2)
                        global_v_idx = (b_row_off + j)*5 + (b_col_off + k)
                        V[global_v_idx, r] += val
                # Add w
                for local_idx, val in enumerate(w_M[m_idx]):
                    if val != 0:
                        k, i = divmod(local_idx, 2)
                        global_w_idx = (b_col_off + k)*2 + i
                        W[global_w_idx, r] += val
            
            return rank_off + 7
        
        def add_2x2x1_block(a_row_off, a_col_off, b_row_off, b_col_off, rank_off):
            # 2x2x1: A is 2x2, B is 2x1, C is 2x1.
            # C0 = A00*B0 + A01*B1
            # C1 = A10*B0 + A11*B1
            # Rank 4.
            # M1: A00*B0 -> u=[1,0,0,0], v=[1,0], w=[1,0]
            # M2: A01*B1 -> u=[0,1,0,0], v=[0,1], w=[1,0]
            # M3: A10*B0 -> u=[0,0,1,0], v=[1,0], w=[0,1]
            # M4: A11*B1 -> u=[0,0,0,1], v=[0,1], w=[0,1]
            
            u_M = [
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]
            ]
            v_M = [
                [1,0],
                [0,1],
                [1,0],
                [0,1]
            ]
            w_M = [
                [1,0],
                [1,0],
                [0,1],
                [0,1]
            ]
            
            for m_idx in range(4):
                r = rank_off + m_idx
                for local_idx, val in enumerate(u_M[m_idx]):
                    if val != 0:
                        i, j = divmod(local_idx, 2)
                        global_u_idx = i*4 + (a_col_off + j)
                        U[global_u_idx, r] += val
                for local_idx, val in enumerate(v_M[m_idx]):
                    if val != 0:
                        j, k = divmod(local_idx, 1) # k is always 0 for 2x1
                        global_v_idx = (b_row_off + j)*5 + (b_col_off + k)
                        V[global_v_idx, r] += val
                for local_idx, val in enumerate(w_M[m_idx]):
                    if val != 0:
                        k, i = divmod(local_idx, 1) # k is always 0
                        global_w_idx = (b_col_off + k)*2 + i
                        W[global_w_idx, r] += val
            
            return rank_off + 4
        
        r = 0
        # Block 1: A_left (col 0) * B_top (row 0, col 0) -> 2x2x2
        r = add_2x2x2_block(0, 0, 0, 0, r)
        # Block 2: A_left (col 0) * B_top (row 0, col 2) -> 2x2x2
        r = add_2x2x2_block(0, 0, 0, 2, r)
        # Block 3: A_left (col 0) * B_top (row 0, col 4) -> 2x2x1
        r = add_2x2x1_block(0, 0, 0, 4, r)
        # Block 4: A_right (col 2) * B_bottom (row 2, col 0) -> 2x2x2
        r = add_2x2x2_block(0, 2, 2, 0, r)
        # Block 5: A_right (col 2) * B_bottom (row 2, col 2) -> 2x2x2
        r = add_2x2x2_block(0, 2, 2, 2, r)
        # Block 6: A_right (col 2) * B_bottom (row 2, col 4) -> 2x2x1
        r = add_2x2x1_block(0, 2, 2, 4, r)
        
        assert r == 36
        return U, V, W, 36
    
    U36, V36, W36, rank36 = construct_rank_36()
    
    if check_decomposition(U36, V36, W36):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # Fallback to schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

BET: {mechanism: "Explicit Strassen-based decomposition for 2x4x5 reduces rank from 40 to 36", q: 0.95, delta: [4, 4]}
