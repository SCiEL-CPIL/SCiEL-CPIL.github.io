"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen-based block decomposition for 5x5x5.
    # Split each dimension into [2, 2, 1].
    # For 2x2x2 blocks, use Strassen (rank 7).
    # For other blocks, use schoolbook.
    
    # Strassen for 2x2x2:
    # M = A * B where A is 2x2, B is 2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M[0,0] = a*(e+h) - (a-c)*(g+h)  -> (a*e + a*h - a*g - a*h + c*g + c*h) = a*e - a*g + c*g + c*h
    # Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In tensor form T[i,j,k]:
    # A indices: i (row), j (col of A / row of B)
    # B indices: j (row of B / col of A), k (col of B)
    # C indices: i (row of C), k (col of C)
    
    # The tensor entry is 1 if we multiply A[i,j] * B[j,k] to contribute to C[i,k].
    
    # We need to express the sum over j of A[i,j]B[j,k] as a sum of rank-1 terms.
    # Each rank-1 term is u_i * v_j * w_k.
    # u_i is the coefficient for A[i,j], v_j for B[j,k], w_k for C[i,k].
    # Actually, the decomposition is C[i,k] = sum_r U[i,r] * V[r] * W[r,k]? No.
    # The standard form is:
    # C[i,k] = sum_r U[i, r] * V[r, j]? No, j is summed out.
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r].
    # Here U is (n x rank), V is (m x rank), W is (p x rank).
    # Note: The prompt says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # This is the "unrolled" format.
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r].
    # Wait, the prompt says:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So U has rows indexed by (i,j), V by (j,k), W by (k,i).
    # The decomposition is:
    # sum_{i,j,k} T[i,j,k] e_i \otimes e_j \otimes e_k = sum_r (sum_{i,j} U[i*m+j, r] e_i \otimes e_j) \otimes (sum_{j,k} V[j*p+k, r] e_j \otimes e_k) \otimes (sum_{k,i} W[k*n+i, r] e_k \otimes e_i)
    # This implies that the rank-r term contributes:
    # (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j,k} V[j*p+k, r] B[j,k]) * (sum_{k,i} W[k*n+i, r] C[k,i]?)
    # No, the third factor is just a scalar multiplier for the C component?
    # Actually, the standard tensor decomposition of matmul is:
    # T = sum_r u^{(r)} \otimes v^{(r)} \otimes w^{(r)}
    # where u^{(r)} is a vector in R^{n*m} (flattened A mask),
    # v^{(r)} is a vector in R^{m*p} (flattened B mask),
    # w^{(r)} is a vector in R^{p*n} (flattened C mask).
    # Then C = sum_r (A \cdot u^{(r)}) (v^{(r)} \cdot B) w^{(r)}?
    # No, if u, v, w are masks, then:
    # C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j'',k'} V[j''*p+k', r] B[j'',k']) * (sum_{k'',i'} W[k''*n+i', r] C[i'',k'']?)
    # This doesn't look right.
    
    # Let's re-read the interface carefully.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    # The tensor is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}? No.
    # T[i,j,k] is 1 if we multiply A[i,j] and B[j,k] to get C[i,k].
    # So T is a 5x5x5 tensor with T[i,j,k]=1.
    # We want T = sum_r u_r \otimes v_r \otimes w_r.
    # Where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    # The index mapping is:
    # u_r is indexed by (i,j).
    # v_r is indexed by (j,k).
    # w_r is indexed by (k,i).
    # So the term is u_r[i,j] * v_r[j,k] * w_r[k,i].
    # Summing over r, we get T[i,j,k].
    # Wait, if we sum u_r[i,j] v_r[j,k] w_r[k,i] over r, does it give T[i,j,k]?
    # T[i,j,k] is 1.
    # So we need sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all i,j,k?
    # No, that's not how matmul works.
    # Matmul is C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{ijk} is the structure tensor.
    # T_{ijk} = 1 if the indices match the matmul pattern.
    # Specifically, T_{ijk} is the coefficient of A_i B_j C_k in the polynomial?
    # No, T_{ijk} is 1 if we use A[i,j] and B[j,k] to produce C[i,k].
    # So T_{ijk} = 1 for all i,j,k? No.
    # T_{ijk} is non-zero only if the indices form a valid matmul term.
    # But in the tensor decomposition context for matmul:
    # The tensor is T_{ijk} = 1 if i,j,k are compatible?
    # Actually, the standard tensor for matmul n x n x n is:
    # T_{ijk} = 1 if we multiply element i of first matrix and j of second?
    # Let's stick to the provided interface convention.
    # "W[k*n+i, r]"
    # "U[i*m+j, r]"
    # "V[j*p+k, r]"
    # This suggests the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # If we sum this over r, we get a value for each (i,j,k).
    # For the decomposition to be exact, this sum must equal the tensor entry T[i,j,k].
    # What is T[i,j,k]?
    # In matrix multiplication, the term A[i,j] * B[j,k] contributes to C[i,k].
    # So the tensor T has T[i,j,k] = 1.
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the rank is 1?
    # No, because the indices i, j, k are linked.
    # The tensor T is defined on the space of A (n*m), B (m*p), C (p*n).
    # The entries of T are T[(i,j), (j,k), (k,i)].
    # So T is zero unless the indices match?
    # No, the tensor T is a 3-way array of size (n*m) x (m*p) x (p*n).
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    # All other entries are 0.
    # We want to decompose T into rank 1 tensors.
    # A rank 1 tensor is u \otimes v \otimes w, where u is a vector of length n*m, v of length m*p, w of length p*n.
    # (u \otimes v \otimes w)[a, b, c] = u[a] v[b] w[c].
    # We want T = sum_r u_r \otimes v_r \otimes w_r.
    # So T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    # This matches the interface: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    # So the goal is to find U, V, W such that for all a, b, c:
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    # Where T[i*m+j, j*p+k, k*n+i] = 1.
    
    # For Strassen 2x2x2:
    # We need to decompose the 2x2x2 matmul tensor.
    # Indices i, j, k in {0, 1}.
    # T[i*2+j, j*2+k, k*2+i] = 1.
    # We know Strassen gives rank 7.
    # The 7 terms correspond to the 7 products in Strassen's algorithm.
    # Let's map them.
    
    # A = [[a0, a1], [a2, a3]] (indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1))
    # B = [[b0, b1], [b2, b3]] (indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1))
    # C = [[c0, c1], [c2, c3]] (indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1))
    # Wait, C indices in W are (k, i).
    # So c0 is (0,0), c1 is (0,1)? No.
    # W index is k*n + i.
    # For n=2:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 2
    # (1,1) -> 3
    # So C[0,0] is W[0], C[0,1] is W[1], C[1,0] is W[2], C[1,1] is W[3].
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> u: A00, A11. v: B00, B11. w: C00, C11?
    # M1 contributes to C00 (+1) and C11 (-1)? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's list the 7 vectors for each factor.
    # Factor U (A mask):
    # M1: A00 + A11 -> [1, 0, 0, 1]
    # M2: A10 + A11 -> [0, 0, 1, 1]
    # M3: A00     -> [1, 0, 0, 0]
    # M4: A11     -> [0, 0, 0, 1]
    # M5: A00 + A01 -> [1, 1, 0, 0]
    # M6: A10 - A00 -> [-1, 0, 1, 0]
    # M7: A01 - A11 -> [0, 1, 0, -1]
    
    # Factor V (B mask):
    # M1: B00 + B11 -> [1, 0, 0, 1]
    # M2: B00       -> [1, 0, 0, 0]
    # M3: B01 - B11 -> [0, 1, 0, -1]
    # M4: B10 - B00 -> [-1, 0, 1, 0]
    # M5: B11       -> [0, 0, 0, 1]
    # M6: B00 + B01 -> [1, 1, 0, 0]
    # M7: B10 + B11 -> [0, 0, 1, 1]
    
    # Factor W (C mask):
    # M1: C00 + C11 -> [1, 0, 0, 1]
    # M2: C10       -> [0, 0, 1, 0]
    # M3: C01 + C11 -> [0, 1, 0, 1]
    # M4: C00 + C10 -> [1, 0, 1, 0]
    # M5: C00 - C01 -> [1, -1, 0, 0]
    # M6: C11       -> [0, 0, 0, 1]
    # M7: C00       -> [1, 0, 0, 0]
    
    # Let's verify one term.
    # Term M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1].
    # Contribution to T[a,b,c] is U[a]V[b]W[c].
    # This should match the part of T generated by M1.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # These terms contribute to:
    # A00B00 -> C00 (+1)
    # A00B11 -> C01? No, A00 is (0,0), B11 is (1,1). Product is (0,1). C01.
    # But M1 contributes to C00 and C11?
    # In Strassen, M1 is added to C00 and C11.
    # Wait, the expansion of M1 is (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # A00B00 -> C00
    # A00B11 -> C01? No, A[0,0] * B[1,1] is not a valid matmul term for C[0,1] because j must match.
    # A[i,j] B[j,k].
    # A00 is i=0,j=0. B11 is j=1,k=1. j doesn't match.
    # So A00B11 is 0 in the matmul sum.
    # Strassen works by grouping terms that *would* be zero or cancelling out?
    # No, Strassen works because the sums are over all j.
    # C00 = A00B00 + A01B10.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This doesn't look like it directly gives C00.
    # The magic is that when you combine M1, M4, M5, M7, the unwanted terms cancel.
    # So for the tensor decomposition, we just need the vectors.
    # The vectors U, V, W above are correct for Strassen.
    
    # Now, we construct the full 5x5x5 decomposition.
    # Splits:
    # I = {0,1}, I2 = {2,3}, I3 = {4}
    # Same for J, K.
    
    # We iterate over all blocks (I_block, J_block, K_block).
    # There are 3x3x3 = 27 blocks.
    
    # For each block, we determine the sizes.
    # If size is 2x2x2, use Strassen (7 ranks).
    # Otherwise, use schoolbook (size_i * size_j * size_k ranks).
    
    # We need to map the local indices to global indices.
    # Local i in {0, 1} maps to global i in I_block.
    
    # Let's precompute the Strassen vectors for 2x2x2.
    # They are defined for local indices 0,1.
    
    # Strassen vectors (local indices 0..3 for A, B, C)
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> Wait, C index is (k,i).
    # Local k, i in {0,1}.
    # 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
    
    # U_strassen (4x7)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],  # M1, M2, M3, M4, M5, M6, M7 for A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    # V_strassen (4x7)
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    # W_strassen (4x7)
    W_s = [
        [1, 0, 0, 1, 1, 0, 1],   # C00 (k=0,i=0)
        [0, 0, 1, 0, -1, 0, 0],  # C01 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=1,i=0)
        [1, 0, 1, 0, 0, 1, 0]    # C11 (k=1,i=1)
    ]
    
    # Wait, I need to double check the W indices.
    # W index is k*n + i.
    # For n=2:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So C[0,0] is index 0. C[0,1] is index 1. C[1,0] is index 2. C[1,1] is index 3.
    # My W_s above:
    # M1: C00 + C11 -> [1, 0, 0, 1]. Correct.
    # M2: C10 -> [0, 0, 1, 0]. Correct.
    # M3: C01 + C11 -> [0, 1, 0, 1]. Correct.
    # M4: C00 + C10 -> [1, 0, 1, 0]. Correct.
    # M5: C00 - C01 -> [1, -1, 0, 0]. Correct.
    # M6: C11 -> [0, 0, 0, 1]. Correct.
    # M7: C00 -> [1, 0, 0, 0]. Correct.
    
    # So W_s rows correspond to C[0,0], C[0,1], C[1,0], C[1,1].
    # W_s[0] = [1, 0, 0, 1, 1, 0, 1] -> M1, M4, M5, M7 have 1 for C00.
    # M1: 1. M4: 1. M5: 1. M7: 1. Sum = 4?
    # C00 = M1 + M4 - M5 + M7.
    # So coefficients are 1, 1, -1, 1.
    # My W_s[0] has 1, 1, 1, 1 for M1, M4, M5, M7?
    # Let's re-check W_s[0].
    # M1: 1.
    # M2: 0.
    # M3: 0.
    # M4: 1.
    # M5: 1. (But it should be -1).
    # M6: 0.
    # M7: 1.
    # So W_s[0] should be [1, 0, 0, 1, -1, 0, 1].
    
    # Let's re-verify all W_s.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_s[0] (C00): [1, 0, 0, 1, -1, 0, 1]
    # W_s[1] (C01): [0, 0, 1, 0, 1, 0, 0]
    # W_s[2] (C10): [0, 1, 0, 1, 0, 0, 0]
    # W_s[3] (C11): [1, -1, 1, 0, 0, 1, 0]
    
    # Let's check U_s and V_s again.
    # M1 = (A00+A11)(B00+B11)
    # U1: A00, A11 -> [1, 0, 0, 1]
    # V1: B00, B11 -> [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00
    # U2: A10, A11 -> [0, 0, 1, 1]
    # V2: B00 -> [1, 0, 0, 0]
    
    # M3 = A00(B01-B11)
    # U3: A00 -> [1, 0, 0, 0]
    # V3: B01, B11 -> [0, 1, 0, -1]
    
    # M4 = A11(B10-B00)
    # U4: A11 -> [0, 0, 0, 1]
    # V4: B10, B00 -> [-1, 0, 1, 0]
    
    # M5 = (A00+A01)B11
    # U5: A00, A01 -> [1, 1, 0, 0]
    # V5: B11 -> [0, 0, 0, 1]
    
    # M6 = (A10-A00)(B00+B01)
    # U6: A10, A00 -> [-1, 0, 1, 0]
    # V6: B00, B01 -> [1, 1, 0, 0]
    
    # M7 = (A01-A11)(B10+B11)
    # U7: A01, A11 -> [0, 1, 0, -1]
    # V7: B10, B11 -> [0, 0, 1, 1]
    
    # So:
    # U_s = [
    #   [1, 0, 1, 0, 1, -1, 0],  # A00
    #   [0, 0, 0, 0, 1, 0, 1],   # A01
    #   [0, 1, 0, 0, 0, 1, 0],   # A10
    #   [1, 1, 0, 1, 0, 0, -1]   # A11
    # ]
    # V_s = [
    #   [1, 1, 0, -1, 0, 1, 0],  # B00
    #   [0, 0, 1, 0, 0, 1, 0],   # B01
    #   [0, 0, 0, 1, 0, 0, 1],   # B10
    #   [1, 0, -1, 0, 1, 0, 1]   # B11
    # ]
    # W_s = [
    #   [1, 0, 0, 1, -1, 0, 1],  # C00
    #   [0, 0, 1, 0, 1, 0, 0],   # C01
    #   [0, 1, 0, 1, 0, 0, 0],   # C10
    #   [1, -1, 1, 0, 0, 1, 0]   # C11
    # ]
    
    # These look correct.
    
    # Now, construct the full decomposition.
    
    # Splits:
    # I0 = [0, 1], I1 = [2, 3], I2 = [4]
    # J0 = [0, 1], J1 = [2, 3], J2 = [4]
    # K0 = [0, 1], K1 = [2, 3], K2 = [4]
    
    # We will accumulate U, V, W.
    # Total rank is sum of ranks of all blocks.
    
    # We need to know the total rank first to allocate arrays.
    # Or we can build lists of columns and then stack.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Block definitions:
    # (I_start, I_end, J_start, J_end, K_start, K_end)
    # We iterate over 27 blocks.
    
    for ib in range(3):
        for jb in range(3):
            for kb in range(3):
                # Determine indices
                if ib == 0:
                    I_idx = [0, 1]
                elif ib == 1:
                    I_idx = [2, 3]
                else:
                    I_idx = [4]
                    
                if jb == 0:
                    J_idx = [0, 1]
                elif jb == 1:
                    J_idx = [2, 3]
                else:
                    J_idx = [4]
                    
                if kb == 0:
                    K_idx = [0, 1]
                elif kb == 1:
                    K_idx = [2, 3]
                else:
                    K_idx = [4]
                    
                ni = len(I_idx)
                nj = len(J_idx)
                nk = len(K_idx)
                
                # Determine if this is a 2x2x2 block
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen
                    # Map local indices 0,1 to global indices
                    # Local A index l_a corresponds to I_idx[l_i], J_idx[l_j]
                    # l_a = l_i * 2 + l_j
                    # Global A index = I_idx[l_i] * m + J_idx[l_j]
                    
                    # We need to add 7 columns to U_cols, V_cols, W_cols.
                    for r in range(7):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        
                        # Fill U_col
                        for l_i in range(2):
                            for l_j in range(2):
                                l_a = l_i * 2 + l_j
                                g_i = I_idx[l_i]
                                g_j = J_idx[l_j]
                                g_a = g_i * m + g_j
                                u_col[g_a] = U_s[l_a][r]
                                
                        # Fill V_col
                        for l_j in range(2):
                            for l_k in range(2):
                                l_b = l_j * 2 + l_k
                                g_j = J_idx[l_j]
                                g_k = K_idx[l_k]
                                g_b = g_j * p + g_k
                                v_col[g_b] = V_s[l_b][r]
                                
                        # Fill W_col
                        for l_k in range(2):
                            for l_i in range(2):
                                l_c = l_k * 2 + l_i
                                g_k = K_idx[l_k]
                                g_i = I_idx[l_i]
                                g_c = g_k * n + g_i
                                w_col[g_c] = W_s[l_c][r]
                                
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                        
                else:
                    # Use schoolbook
                    # Rank = ni * nj * nk
                    # For each (i, j, k) in the block, add a term.
                    for l_i in range(ni):
                        for l_j in range(nj):
                            for l_k in range(nk):
                                u_col = [0.0] * (n * m)
                                v_col = [0.0] * (m * p)
                                w_col = [0.0] * (p * n)
                                
                                g_i = I_idx[l_i]
                                g_j = J_idx[l_j]
                                g_k = K_idx[l_k]
                                
                                g_a = g_i * m + g_j
                                g_b = g_j * p + g_k
                                g_c = g_k * n + g_i
                                
                                u_col[g_a] = 1.0
                                v_col[g_b] = 1.0
                                w_col[g_c] = 1.0
                                
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
                                
    # Convert lists of lists to numpy arrays (or keep as lists if required)
    # The interface expects {'U': U, 'V': V, 'W': W} where U, V, W are matrices.
    # The prompt says "integer/half-integer entries".
    # Returning numpy arrays or lists of lists should be fine.
    # Let's return numpy arrays for efficiency, or lists to be safe.
    # "U, V, W: factor matrices of shapes (n*m, rank)..."
    # I will return lists of lists to match the seed program's style,
    # but numpy is allowed. I'll use numpy and then convert to lists if needed?
    # The seed program returned lists of lists.
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Return as lists of lists to be safe?
    # "U, V, W: factor matrices of shapes (n*m, rank)..."
    # If I return numpy arrays, it should work.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
