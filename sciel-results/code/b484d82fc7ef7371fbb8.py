"""Matrix multiplication 4x4x7 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = 4, 4, 7
    
    # Best known: Strassen(4x4x4) + schoolbook(4x4x3) = 49 + 48 = 97
    # Let's build this and then try to improve
    
    # First, build the 4x4x4 Strassen decomposition
    # Strassen for 2x2: rank 7
    # For 4x4: decompose into 4 blocks of 2x2, apply Strassen to each pair
    # Actually, 4x4x4 can use Strassen recursively: split into 2x2 blocks
    # That gives rank 7^2 = 49
    
    # Let's construct Strassen 2x2x2 first (rank 7)
    # C = A*B where A, B are 2x2
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A01)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 4x4x4, we split A into 4 blocks A_ij (i,j in {0,1}), each 2x2
    # B into 4 blocks B_jk (j,k in {0,1}), each 2x2
    # C_ik = sum_j A_ij * B_jk
    # Apply Strassen to the 2x2 "block" multiplication where each element is a 2x2 matrix
    # This requires 7 multiplications of 2x2 matrices, each of which can be done with rank 7
    # Total rank = 7 * 7 = 49
    
    # Let me build this explicitly
    
    # Strassen 2x2x2 factors (U, V, W) where U is 4x7, V is 4x7, W is 4x7
    # Indexing: U[row, col], V[row, col], W[row, col]
    # For 2x2: A has indices (i,j) -> row i*2+j, B has (j,k) -> row j*2+k, C has (k,i) -> row k*2+i
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen, each 4x7."""
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00 += M1
        W[3, 0] = 1  # C11 += M1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10 += M2
        W[3, 1] = -1 # C11 -= M2
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1  # A00
        V[1, 2] = 1  # B01
        V[3, 2] = -1 # -B11
        W[1, 2] = 1  # C01 += M3
        W[3, 2] = 1  # C11 += M3
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1  # A11
        V[2, 3] = 1  # B10
        V[0, 3] = -1 # -B00
        W[0, 3] = 1  # C00 += M4
        W[2, 3] = 1  # C10 += M4
        
        # M5 = (A00+A10)*B11
        U[0, 4] = 1  # A00
        U[2, 4] = 1  # A10
        V[3, 4] = 1  # B11
        W[0, 4] = -1 # C00 -= M5
        W[1, 4] = 1  # C01 += M5
        
        # M6 = (A11-A01)*(B00+B10)
        U[3, 5] = 1  # A11
        U[1, 5] = -1 # -A01
        V[0, 5] = 1  # B00
        V[2, 5] = 1  # B10
        W[3, 5] = 1  # C11 += M6
        
        # M7 = (A01-A11)*(B01+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1 # -A11
        V[1, 6] = 1  # B01
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00 += M7
        W[1, 6] = 1  # C01 += M7
        
        return U, V, W
    
    U2, V2, W2 = strassen_2x2()
    
    # Now for 4x4x4: split into 2x2 blocks
    # A is 4x4, split into A00, A01, A10, A11 each 2x2
    # B is 4x4, split into B00, B01, B10, B11 each 2x2
    # C is 4x4, split into C00, C01, C10, C11 each 2x2
    # C_ik = sum_j A_ij * B_jk
    # Use Strassen on the "block" level:
    # M1_block = (A00+A11)*(B00+B11)
    # M2_block = (A10+A11)*B00
    # M3_block = A00*(B01-B11)
    # M4_block = A11*(B10-B00)
    # M5_block = (A00+A10)*B11
    # M6_block = (A11-A01)*(B00+B10)
    # M7_block = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each block multiplication (2x2 * 2x2) has rank 7
    # So each M_block has rank 7
    # Total rank = 7 * 7 = 49
    
    # Let's build the 4x4x4 decomposition
    # A indices: A[i,j] where i,j in 0..3. Block A_ab has i=2a+i', j=2b+j'
    # So A[2a+i', 2b+j'] for a,b in {0,1}, i',j' in {0,1}
    
    U4 = np.zeros((16, 49), dtype=int)  # 4*4 = 16
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # For each Strassen term t (0..6) and each inner Strassen term s (0..6):
    # The column index is t*7 + s
    
    # Block positions in 4x4:
    # A00: rows 0-3, cols 0-3 -> A[0:4, 0:4]
    # A01: rows 0-3, cols 4-7 -> but wait, A is 4x4, so A01 is A[0:2, 2:4]
    # Let me be more careful.
    
    # A is 4x4. Split into 2x2 blocks:
    # A00 = A[0:2, 0:2] -> indices (0,0),(0,1),(1,0),(1,1) -> flat 0,1,2,3
    # A01 = A[0:2, 2:4] -> indices (0,2),(0,3),(1,2),(1,3) -> flat 2,3,6,7
    # A10 = A[2:4, 0:2] -> indices (2,0),(2,1),(3,0),(3,1) -> flat 4,5,8,9
    # A11 = A[2:4, 2:4] -> indices (2,2),(2,3),(3,2),(3,3) -> flat 6,7,10,11
    
    # B is 4x4:
    # B00 = B[0:2, 0:2] -> flat 0,1,2,3
    # B01 = B[0:2, 2:4] -> flat 2,3,6,7
    # B10 = B[2:4, 0:2] -> flat 4,5,8,9
    # B11 = B[2:4, 2:4] -> flat 6,7,10,11
    
    # C is 4x4:
    # C00 = C[0:2, 0:2] -> flat 0,1,2,3
    # C01 = C[0:2, 2:4] -> flat 2,3,6,7
    # C10 = C[2:4, 0:2] -> flat 4,5,8,9
    # C11 = C[2:4, 2:4] -> flat 6,7,10,11
    
    # For each Strassen term t, we have a linear combination of blocks
    # Then for each inner Strassen term s, we multiply the resulting 2x2 block
    
    # Let me define the block linear combinations for each Strassen term:
    # t=0 (M1): A: A00+A11, B: B00+B11
    # t=1 (M2): A: A10+A11, B: B00
    # t=2 (M3): A: A00, B: B01-B11
    # t=3 (M4): A: A11, B: B10-B00
    # t=4 (M5): A: A00+A10, B: B11
    # t=5 (M6): A: A11-A01, B: B00+B10
    # t=6 (M7): A: A01-A11, B: B01+B11
    
    # A block coefficients for each t:
    A_block_coeffs = [
        {0: 1, 3: 1},    # t=0: A00+A11
        {2: 1, 3: 1},    # t=1: A10+A11
        {0: 1},           # t=2: A00
        {3: 1},           # t=3: A11
        {0: 1, 2: 1},    # t=4: A00+A10
        {3: 1, 1: -1},   # t=5: A11-A01
        {1: 1, 3: -1},   # t=6: A01-A11
    ]
    
    # B block coefficients for each t:
    B_block_coeffs = [
        {0: 1, 3: 1},    # t=0: B00+B11
        {0: 1},           # t=1: B00
        {1: 1, 3: -1},   # t=2: B01-B11
        {2: 1, 0: -1},   # t=3: B10-B00
        {3: 1},           # t=4: B11
        {0: 1, 2: 1},    # t=5: B00+B10
        {1: 1, 3: 1},    # t=6: B01+B11
    ]
    
    # C block coefficients for each t:
    # C00 = M1+M4-M5+M7 -> t=0:+1, t=3:+1, t=4:-1, t=6:+1
    # C01 = M3+M5 -> t=2:+1, t=4:+1
    # C10 = M2+M4 -> t=1:+1, t=3:+1
    # C11 = M1-M2+M3+M6 -> t=0:+1, t=1:-1, t=2:+1, t=5:+1
    C_block_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # C00
        {2: 1, 4: 1},                  # C01
        {1: 1, 3: 1},                  # C10
        {0: 1, 1: -1, 2: 1, 5: 1},    # C11
    ]
    
    # Block to flat index mapping for 2x2 blocks in 4x4:
    # Block (a,b) where a,b in {0,1}: rows 2a to 2a+1, cols 2b to 2b+1
    # Flat index of A[2a+i, 2b+j] = (2a+i)*4 + (2b+j)
    
    def block_flat_indices(block_a, block_b):
        """Return list of flat indices for block (block_a, block_b) in 4x4."""
        indices = []
        for i in range(2):
            for j in range(2):
                row = 2 * block_a + i
                col = 2 * block_b + j
                indices.append(row * 4 + col)
        return indices
    
    # For each Strassen term t and inner term s:
    col = 0
    for t in range(7):
        for s in range(7):
            # U contribution: for each A block in A_block_coeffs[t], 
            #   the inner Strassen U2[s] gives coefficients for the 2x2 block
            for block_a, coeff_a in A_block_coeffs[t].items():
                flat_idx = block_flat_indices(block_a, 0)  # This is wrong, need to think about this differently
                # Actually, A_block_coeffs[t] gives which A blocks are involved
                # But each A block is 2x2, and the inner Strassen operates on the 2x2
                # So for A block (block_a, ?), we need to know which column block
                # Wait, I'm overcomplicating this.
                
                # Let me restart the 4x4x4 construction more carefully.
            col += 1
    col = 0
    
    # Let me try a different approach: directly construct by iterating
    # over all (i,j,k) and using the Strassen structure.
    
    # Actually, let me just build the 4x4x4 tensor and use the known rank-49 decomposition.
    # The 4x4x4 multiplication tensor T[i,j,k] = 1 if we're computing C[k,i] from A[i,j]*B[j,k]
    
    # For 4x4x4: T has shape (16, 16, 16) in the U,V,W convention
    # U is 16x49, V is 16x49, W is 16x49
    
    # Let me construct it by Kronecker product of two 2x2x2 Strassen decompositions.
    # A = A1 ⊗ A2, B = B1 ⊗ B2, C = C1 ⊗ C2 where C1 = A1*B1, C2 = A2*B2
    # But that's not quite right for matrix multiplication.
    
    # Actually, for (A⊗B)(C⊗D) = (AC)⊗(BD), so 4x4x4 can be seen as
    # two independent 2x2x2 multiplications, giving rank 7+7=14, but that's wrong
    # because the blocks interact.
    
    # Let me just do the explicit block Strassen construction.
    
    # A is 4x4, index as A[i,j] for i,j in 0..3
    # Split into 2x2 blocks: A[2a+i', 2b+j'] for a,b,i',j' in {0,1}
    # Let me denote A_ab[i',j'] = A[2a+i', 2b+j']
    
    # Strassen at block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each M_t is a 2x2 matrix = sum of products of 2x2 blocks
    # Each such product can be done with rank 7 (Strassen 2x2)
    
    # So for each t, M_t = sum over inner Strassen terms
    # Total rank = 7 * 7 = 49
    
    # Let me build U4, V4, W4 explicitly.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # For each t (0..6) and s (0..6):
    col = 0
    for t in range(7):
        for s in range(7):
            # Inner Strassen term s contributes to U2, V2, W2
            # Outer Strassen term t determines which blocks
            
            # U contribution: for each (i',j') in 2x2, U2[i'*2+j', s] gives coefficient
            # for A[i',j'] in the inner product
            # But we need to map to the correct block
            
            # A blocks involved in term t:
            for block_a, coeff_a in A_block_coeffs[t].items():
                # A_block_coeffs[t] is a dict {block_index: coeff}
                # block_index: 0=A00, 1=A01, 2=A10, 3=A11
                # For each inner Strassen component:
                for i_prime in range(2):
                    for j_prime in range(2):
                        # A[i',j'] in block block_a
                        # flat index: (2*block_a_row + i_prime)*4 + (2*block_a_col + j_prime)
                        # block_a: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                        ba_row = block_a // 2
                        ba_col = block_a % 2
                        flat_i = 2 * ba_row + i_prime
                        flat_j = 2 * ba_col + j_prime
                        flat_idx = flat_i * 4 + flat_j
                        U4[flat_idx, col] += coeff_a * U2[i_prime * 2 + j_prime, s]
            
            # V contribution:
            for block_b, coeff_b in B_block_coeffs[t].items():
                bb_row = block_b // 2
                bb_col = block_b % 2
                for i_prime in range(2):
                    for j_prime in range(2):
                        flat_i = 2 * bb_row + i_prime
                        flat_j = 2 * bb_col + j_prime
                        flat_idx = flat_i * 4 + flat_j
                        V4[flat_idx, col] += coeff_b * V2[i_prime * 2 + j_prime, s]
            
            # W contribution:
            for block_c, coeff_c in C_block_coeffs[t].items():
                bc_row = block_c // 2
                bc_col = block_c % 2
                for i_prime in range(2):
                    for j_prime in range(2):
                        flat_i = 2 * bc_row + i_prime
                        flat_j = 2 * bc_col + j_prime
                        flat_idx = flat_i * 4 + flat_j
                        W4[flat_idx, col] += coeff_c * W2[i_prime * 2 + j_prime, s]
            
            col += 1
    
    # Now build 4x4x3 schoolbook (rank 48)
    # C is 4x3, A is 4x4, B is 4x3
    # U is 16x48, V is 12x48, W is 12x48
    # W[k*n+i, c] = W[k*4+i, c] for k in 0..2, i in 0..3
    
    U43 = np.zeros((16, 48), dtype=int)
    V43 = np.zeros((12, 48), dtype=int)
    W43 = np.zeros((12, 48), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U43[i * 4 + j, col] = 1
                V43[j * 3 + k, col] = 1
                W43[k * 4 + i, col] = 1
                col += 1
    
    # Combine: total rank = 49 + 48 = 97
    # U is 16x97, V is 28x97 (16+12), W is 28x97 (16+12)
    # Wait, V for 4x4x4 is 16x49, V for 4x4x3 is 12x48
    # But V should be (m*p, rank) = (4*7, rank) = (28, rank)
    # For the 4x4x4 part, V is 16x49 (for the first 4 columns of B)
    # For the 4x4x3 part, V is 12x48 (for the last 3 columns of B)
    # So V_total is 28x97: first 16 rows from V4, next 12 rows from V43
    
    U_total = np.zeros((16, 97), dtype=int)
    V_total = np.zeros((28, 97), dtype=int)
    W_total = np.zeros((28, 97), dtype=int)
    
    U_total[:, 0:49] = U4
    U_total[:, 49:97] = U43
    V_total[0:16, 0:49] = V4
    V_total[16:28, 49:97] = V43
    W_total[0:16, 0:49] = W4
    W_total[16:28, 49:97] = W43
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c]
    # Wait, the indexing convention:
    # U is (n*m, rank) = (16, 97), indexed by i*m+j = i*4+j
    # V is (m*p, rank) = (28, 97), indexed by j*p+k = j*7+k
    # W is (p*n, rank) = (28, 97), indexed by k*n+i = k*4+i
    
    # For the 4x4x4 part (k in 0..3):
    # V4 is indexed by j*4+k, but we need j*7+k
    # So V4[j*4+k] should go to V_total[j*7+k] for k in 0..3
    # Similarly, W4[k*4+i] should go to W_total[k*4+i] for k in 0..3
    
    # Let me redo this with correct indexing.
    
    U_total = np.zeros((16, 97), dtype=int)
    V_total = np.zeros((28, 97), dtype=int)
    W_total = np.zeros((28, 97), dtype=int)
    
    # 4x4x4 part (columns 0..3 of B, i.e., k in 0..3)
    U_total[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V_total[j * 7 + k, 0:49] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W_total[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
    
    # 4x4x3 part (columns 4..6 of B, i.e., k in 4..6)
    U_total[:, 49:97] = U43
    for j in range(4):
        for k in range(3):
            V_total[j * 7 + (4 + k), 49:97] = V43[j * 3 + k, 49:97]
    for k in range(3):
        for i in range(4):
            W_total[(4 + k) * 4 + i, 49:97] = W43[k * 4 + i, 49:97]
    
    # Verify
    error = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                expected = 1.0
                actual = 0.0
                for c in range(97):
                    actual += U_total[i * 4 + j, c] * V_total[j * 7 + k, c] * W_total[k * 4 + i, c]
                if abs(actual - expected) > 1e-9:
                    error += 1
    
    if error > 0:
        # Fallback to schoolbook
        rank = 112
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((28, rank), dtype=int)
        W = np.zeros((28, rank), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i * 4 + j, col] = 1
                    V[j * 7 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        U_total = U
        V_total = V
        W_total = W
    
    # Try to reduce rank by zeroing out some columns if possible
    # (This is a heuristic; in general, we can't just zero out columns)
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
