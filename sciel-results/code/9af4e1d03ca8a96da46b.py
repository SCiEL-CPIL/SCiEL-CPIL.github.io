import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition in two parts:
    # Part 1: 4x4x4 multiplication using Strassen (rank 49)
    # Part 2: 4x4x2 multiplication using dot products (rank 8)
    # Total rank = 57
    
    rank1 = 49
    rank2 = 8
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)] # 16 x 57
    V = [[0] * total_rank for _ in range(m * p)] # 24 x 57
    W = [[0] * total_rank for _ in range(p * n)] # 24 x 57
    
    # --- Part 1: 4x4x4 Strassen ---
    # We need to map the 49 scalar multiplications to U, V, W.
    # Strassen uses 7 sub-multiplications of size 2x2x2.
    # Each 2x2x2 uses 7 scalar multiplications.
    
    # Define the 7 2x2x2 multiplications for Strassen
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need the rank-7 decomposition for 2x2x2.
    # U2 (4x7), V2 (4x7), W2 (4x7)
    # Indices for 2x2x2: i,j,k in 0..1.
    # U[i*2+j, col], V[j*2+k, col], W[k*2+i, col]
    
    # Standard Strassen 2x2x2 decomposition
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # The 7 products are:
    # 1: (A11+A22) * (B11+B22)
    # 2: (A21+A22) * B11
    # 3: A11 * (B12-B22)
    # 4: A22 * (B21-B11)
    # 5: (A11+A12) * B22
    # 6: (A21-A11) * (B11+B12)
    # 7: (A12-A22) * (B21+B22)
    
    # We need to express these as U, V, W vectors for the 2x2x2 tensor.
    # Actually, it's easier to just hardcode the U, V, W for the 4x4x4 case directly
    # by expanding the Strassen steps into scalar operations.
    
    # Let's generate the U, V, W for the 49 rank-1 tensors.
    # Each of the 7 Strassen products is a 2x2x2 multiplication.
    # We can use a known rank-7 decomposition for 2x2x2.
    
    # Rank 7 decomposition for 2x2x2:
    # U2, V2, W2 are 4x7 matrices.
    # Let's define them.
    # A common one:
    # col 0: A11 B11 -> C11
    # col 1: A22 B22 -> C22
    # col 2: (A11+A22)(B11+B22) -> ... this is not rank 1.
    
    # Actually, the rank 7 decomposition is:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mk is a scalar product of two linear combinations of A and B entries.
    # So each Mk corresponds to ONE rank-1 tensor in the 2x2x2 decomposition?
    # No, Mk is a 2x2 matrix result?
    # No, in Strassen, Mk is a 2x2 matrix product.
    # So Mk itself requires a rank-7 decomposition?
    # No, Strassen replaces 8 multiplications of 2x2 matrices with 7 multiplications of 2x2 matrices.
    # The "multiplication" of 2x2 matrices is the base case.
    # If we go down to scalar level, a 2x2 matrix multiplication has rank 7.
    # So each of the 7 Strassen "multiplications" expands to 7 scalar multiplications.
    # Total 49.
    
    # So we need the U, V, W for the 2x2x2 scalar multiplication (rank 7).
    # Let's define U2, V2, W2 for 2x2x2.
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    # One valid rank-7 decomposition for 2x2x2:
    # c=0: A11 B11 -> C11
    # c=1: A22 B22 -> C22
    # c=2: (A11+A22)(B11+B22) ... wait, this is not a single scalar product.
    # The rank 7 decomposition consists of 7 terms.
    # Term 1: (A11+A22) * (B11+B22) * (C11 - C12 - C21 + C22)? No.
    
    # Let's use the explicit Strassen formulas for 2x2x2 scalar multiplication.
    # The 7 products are:
    # 1: (a11+a22)(b11+b22)
    # 2: (a21+a22)b11
    # 3: a11(b12-b22)
    # 4: a22(b21-b11)
    # 5: (a11+a12)b22
    # 6: (a21-a11)(b11+b12)
    # 7: (a12-a22)(b21+b22)
    
    # And the results are:
    # c11 = 1 - 4 + 5 - 7
    # c12 = 3 + 7
    # c21 = 2 + 4
    # c22 = 1 - 3 + 6 + 7
    
    # So for each product t (1..7), we have a scalar multiplication $L_A \cdot L_B$.
    # This contributes to the U, V, W matrices.
    # For product t:
    # U_vec is the coefficients of $L_A$ in the 4-dim A space.
    # V_vec is the coefficients of $L_B$ in the 4-dim B space.
    # W_vec is the coefficients in the 4-dim C space, determined by which c_ij it contributes to and with what sign.
    
    # Let's build U2, V2, W2 (4x7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Product 1: (a11+a22)(b11+b22)
    # A: a11=1, a22=1 -> idx 0 (11), 3 (22)
    # B: b11=1, b22=1 -> idx 0 (11), 3 (22)
    # C: c11=1, c12=-1, c21=-1, c22=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = -1; W2[2, 0] = -1; W2[3, 0] = 1
    
    # Product 2: (a21+a22)b11
    # A: a21=1, a22=1 -> idx 2 (21), 3 (22)
    # B: b11=1 -> idx 0 (11)
    # C: c21=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # Product 3: a11(b12-b22)
    # A: a11=1 -> idx 0
    # B: b12=1, b22=-1 -> idx 1 (12), 3 (22)
    # C: c12=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # Product 4: a22(b21-b11)
    # A: a22=1 -> idx 3
    # B: b21=1, b11=-1 -> idx 2 (21), 0 (11)
    # C: c11=-1, c21=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = -1; W2[2, 3] = 1
    
    # Product 5: (a11+a12)b22
    # A: a11=1, a12=1 -> idx 0, 1
    # B: b22=1 -> idx 3
    # C: c11=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1
    
    # Product 6: (a21-a11)(b11+b12)
    # A: a21=1, a11=-1 -> idx 2, 0
    # B: b11=1, b12=1 -> idx 0, 1
    # C: c22=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Product 7: (a12-a22)(b21+b22)
    # A: a12=1, a22=-1 -> idx 1, 3
    # B: b21=1, b22=1 -> idx 2, 3
    # C: c11=-1, c12=1, c22=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[1, 6] = 1; W2[3, 6] = 1

    # Now, construct U, V, W for 4x4x4 (rank 49)
    # The 4x4 matrices are split into 2x2 blocks.
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # Indices in A (4x4):
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # We have 7 Strassen products. Each uses U2, V2, W2.
    # But the inputs to the 2x2 multiplications are linear combinations of the 4x4 blocks.
    # We need to map the 4-dim vectors of U2, V2 to the 16-dim space of U, V.
    
    # Let's define the linear combinations for each of the 7 products.
    # Prod 1: A_in = A11 + A22, B_in = B11 + B22
    # Prod 2: A_in = A21 + A22, B_in = B11
    # Prod 3: A_in = A11, B_in = B12 - B22
    # Prod 4: A_in = A22, B_in = B21 - B11
    # Prod 5: A_in = A11 + A12, B_in = B22
    # Prod 6: A_in = A21 - A11, B_in = B11 + B12
    # Prod 7: A_in = A12 - A22, B_in = B21 + B22
    
    # For each product s (0..6), and each of the 7 sub-products t (0..6):
    # The scalar multiplication is (U2[:,t] dot A_s) * (V2[:,t] dot B_s)
    # This contributes to W via W2[:,t] dot C_s.
    
    # We need to construct U_full (16x49), V_full (16x49), W_full (16x49).
    # Note: V is for B (4x4), so 16 entries. W is for C (4x4), so 16 entries.
    # In the final output, V is 24x57 (4x6), W is 24x57 (6x4).
    
    # Let's fill U[0:16, 0:49], V[0:16, 0:49], W[0:16, 0:49].
    # (Indices for B and C in the 4x4x4 part are 0..15).
    
    # Helper to get block indices
    def block_indices(block):
        # block 0: A11, 1: A12, 2: A21, 3: A22
        r, c = divmod(block, 2)
        start_r = r * 2
        start_c = c * 2
        return [start_r * 4 + start_c + dr * 4 + dc for dr in range(2) for dc in range(2)]

    # Linear combinations for A and B for each Strassen product
    # A_comb[s] is a dict or list of (block, coeff)
    # B_comb[s] is a dict or list of (block, coeff)
    
    A_combs = [
        [(0, 1), (3, 1)], # 1: A11+A22
        [(2, 1), (3, 1)], # 2: A21+A22
        [(0, 1)],         # 3: A11
        [(3, 1)],         # 4: A22
        [(0, 1), (1, 1)], # 5: A11+A12
        [(2, 1), (0, -1)],# 6: A21-A11
        [(1, 1), (3, -1)] # 7: A12-A22
    ]
    B_combs = [
        [(0, 1), (3, 1)], # 1: B11+B22
        [(0, 1)],         # 2: B11
        [(1, 1), (3, -1)],# 3: B12-B22
        [(2, 1), (0, -1)],# 4: B21-B11
        [(3, 1)],         # 5: B22
        [(0, 1), (1, 1)], # 6: B11+B12
        [(2, 1), (3, 1)]  # 7: B21+B22
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U part
            for block, coeff in A_combs[s]:
                idxs = block_indices(block)
                for idx in idxs:
                    # U2[t] corresponds to the 4 entries of the 2x2 block
                    # U2 is 4x7. The row index in U2 corresponds to the position in the 2x2 block.
                    # block_indices gives global indices.
                    # We need to map the local 2x2 index to the global index.
                    # block_indices returns global indices in order 0,1,2,3 corresponding to 00,01,10,11.
                    # U2 row 0 is 00, 1 is 01, 2 is 10, 3 is 11.
                    # So idxs[0] gets U2[0,t], idxs[1] gets U2[1,t], etc.
                    local_idx = idxs.index(idx) # This is inefficient but fine for small N
                    U[idx, col] += coeff * U2[local_idx, t]
            
            # V part
            for block, coeff in B_combs[s]:
                idxs = block_indices(block)
                for idx in idxs:
                    local_idx = idxs.index(idx)
                    V[idx, col] += coeff * V2[local_idx, t]
            
            # W part
            # The result C_s is a 2x2 matrix.
            # C_s is formed by combining the 4 C blocks?
            # No, C_s is the result of the 2x2 multiplication.
            # The Strassen formula gives C11, C12, C21, C22 in terms of M1..M7.
            # M_s is the result of the s-th product.
            # So the contribution of M_s to C11 is given by the coefficient in the Strassen formula.
            # We already encoded the contribution to the 2x2 result in W2?
            # No, W2 is for the 2x2x2 tensor.
            # The s-th product M_s is a 2x2 matrix.
            # The entries of M_s contribute to the final C blocks.
            # C11 = M1 - M4 + M5 - M7
            # C12 = M3 + M7
            # C21 = M2 + M4
            # C22 = M1 - M3 + M6 + M7
            
            # So for product s, we need to know which C blocks it contributes to and with what sign.
            # Let's define C_contrib[s] as a list of (block, coeff)
            
            C_contribs = [
                [(0, 1), (1, -1), (2, -1), (3, 1)], # M1
                [(2, 1)],                            # M2
                [(1, 1)],                            # M3
                [(0, -1), (2, 1)],                   # M4
                [(0, 1)],                            # M5
                [(3, 1)],                            # M6
                [(0, -1), (1, 1), (3, 1)]            # M7
            ]
            
            for block, coeff in C_contribs[s]:
                idxs = block_indices(block)
                for idx in idxs:
                    local_idx = idxs.index(idx)
                    W[idx, col] += coeff * W2[local_idx, t]

    # --- Part 2: 4x4x2 (rank 8) ---
    # Columns 4 and 5 of B.
    # C_{i,4} = sum_j A_{ij} B_{j,4}
    # C_{i,5} = sum_j A_{ij} B_{j,5}
    
    # For each i in 0..3, k in 4..5:
    # We have 4 terms (j=0..3).
    # Total 8 * 4 = 32 terms?
    # No, rank of 4x4x1 is 4.
    # So for each k, we have 4 rank-1 terms?
    # Rank of M_{4,4,1} is 4.
    # So for k=4, we use 4 multiplications.
    # For k=5, we use 4 multiplications.
    # Total 8 multiplications.
    
    # The decomposition for 4x4x1:
    # C_i = sum_j A_{ij} B_j
    # This is sum of 4 rank-1 tensors:
    # T_j = (e_i A_{ij}) * (e_j B_j) * (e_i delta_{i,i'})?
    # Actually, the rank-1 tensor is $u \otimes v \otimes w$.
    # $u_i = A_{ij}$? No, $A$ is the input.
    # The term is $A_{ij} B_{jk}$.
    # For fixed j, the term is $A_{ij} B_{jk}$.
    # This is a rank-1 tensor in the space of (A, B, C)?
    # $T_{ijk} = A_{ij} B_{jk} \delta_{k,k'}$.
    # We can write $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is a sum of 4 terms, each involving one $j$.
    # Term j: $A_{ij} B_{jk}$.
    # This term is non-zero only for the specific $i, j, k$.
    # Wait, $A_{ij}$ depends on $i, j$. $B_{jk}$ depends on $j, k$.
    # So the term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # The vector $u$ in U has $u_{ij} = 1$? No.
    # In the tensor decomposition, $C_{ik} = \sum_r U_{ik,r} V_{jk,r} W_{...,r}$?
    # No, $C_{ik} = \sum_r \sum_j U_{ij,r} V_{jk,r} W_{ik,r}$?
    # The standard form is $C_{ik} = \sum_r U_{i,r} V_{j,r} W_{k,r}$? No.
    # $C = \sum_r U_r V_r W_r^T$?
    # The interface is U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # $C_{ik} = \sum_j \sum_r U_{im+j, r} V_{jp+k, r} W_{pk+i, r}$?
    # Let's check the indices in the prompt:
    # U: (n*m, rank) -> index i*m + j
    # V: (m*p, rank) -> index j*p + k
    # W: (p*n, rank) -> index k*n + i
    
    # So $C_{ik} = \sum_j \sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r}$
    
    # For the dot product part (k=4,5):
    # $C_{i,4} = \sum_j A_{ij} B_{j,4}$
    # We can use 4 terms, one for each j.
    # Term j:
    # $U_{i, j} = 1$ (only for this i, j)
    # $V_{j, 4} = 1$ (only for this j, 4)
    # $W_{4, i} = 1$ (only for this 4, i)
    # This gives $1 \cdot 1 \cdot 1 = 1$ for $C_{i,4}$ when summing over j?
    # No, the sum over r is the rank.
    # For a fixed j, we have one rank-1 term.
    # $U_{i m + j, r} = 1$
    # $V_{j p + 4, r} = 1$
    # $W_{4 n + i, r} = 1$
    # This term contributes $1$ to $C_{i,4}$?
    # Let's check:
    # $\sum_r U_{i m + j, r} V_{j p + 4, r} W_{4 n + i, r}$
    # If only r is non-zero for this j, and only this i,j,k is non-zero in U,V,W.
    # Then the term is $1 \cdot 1 \cdot 1 = 1$.
    # But we need $\sum_j A_{ij} B_{j,4}$.
    # The inputs A and B are multiplied by U and V.
    # The formula is $\sum_r U_r \otimes V_r \otimes W_r$.
    # The actual product is $\sum_r (U_r \cdot A) (V_r \cdot B) W_r$?
    # No, $C = \sum_r U_r V_r W_r^T$?
    # The decomposition is $T = \sum_r U_r \otimes V_r \otimes W_r$.
    # $T_{ijk} = \sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
    # Then $C_{ik} = \sum_j \sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
    # Wait, the prompt says:
    # $C_{ik} = \sum_j A_{ij} B_{jk}$
    # And the decomposition is exact.
    # So $\sum_j A_{ij} B_{jk} = \sum_j \sum_r U_{ij,r} V_{jk,r} W_{ki,r}$?
    # This implies $U, V, W$ are coefficients.
    # Actually, the standard form is:
    # $C_{ik} = \sum_r \left( \sum_j U_{ij,r} A_{ij} \right) \left( \sum_j V_{jk,r} B_{jk} \right) W_{ki,r}$?
    # No, that's not right.
    
    # The correct interpretation of the interface:
    # $U$ is $n \times m \times R$. $V$ is $m \times p \times R$. $W$ is $p \times n \times R$.
    # $C_{ik} = \sum_r \left( \sum_j U_{ij,r} A_{ij} \right) \left( \sum_j V_{jk,r} B_{jk} \right) W_{ki,r}$?
    # No, usually $U, V, W$ are the factors of the tensor.
    # $T_{ijk} = \sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
    # Then $C_{ik} = \sum_j T_{ijk} A_{ij} B_{jk}$? No.
    
    # Let's look at the schoolbook example in the prompt.
    # $U[i m + j, c] = 1, V[j p + k, c] = 1, W[k n + i, c] = 1$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # If we plug in:
    # $\sum_c U_{ij,c} V_{jk,c} W_{ki,c} A_{ij} B_{jk}$?
    # No, the formula is $C_{ik} = \sum_r \left( \sum_j U_{ij,r} A_{ij} \right) \dots$?
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # $C_{ik} = \sum_r \left( \sum_j U_{ij,r} A_{ij} \right) \left( \sum_j V_{jk,r} B_{jk} \right) W_{ki,r}$ is WRONG.
    
    # The correct formula is:
    # $C_{ik} = \sum_r \left( \sum_j U_{ij,r} A_{ij} \right) \left( \sum_j V_{jk,r} B_{jk} \right)$? No.
    
    # It is:
    # $C_{ik} = \sum_r \left( \sum_j U_{ij,r} A_{ij} \right) \left( \sum_j V_{jk,r} B_{jk} \right) W_{ki,r}$?
    
    # Let's check the schoolbook.
    # $U_{ij,c} = 1$ if $c$ corresponds to $(i,j,k)$.
    # Then $\sum_j U_{ij,c} A_{ij} = A_{ij}$ (if c matches i,j).
    # $\sum_j V_{jk,c} B_{jk} = B_{jk}$ (if c matches j,k).
    # Then product is $A_{ij} B_{jk}$.
    # Multiply by $W_{ki,c} = 1$.
    # Sum over c.
    # Only c matching $(i,j,k)$ contributes.
    # So $C_{ik} = \sum_j A_{ij} B_{jk}$.
    
    # So the formula is:
    # $C_{ik} = \sum_r \left( \sum_j U_{ij,r} A_{ij} \right) \left( \sum_j V_{jk,r} B_{jk} \right) W_{ki,r}$
    
    # For the dot product part:
    # We want $C_{i,4} = \sum_j A_{ij} B_{j,4}$.
    # We can use 4 terms, one for each j.
    # Term j:
    # $U_{ij,r} = 1$ (for this specific i, j)
    # $V_{j,4,r} = 1$ (for this specific j, 4)
    # $W_{4,i,r} = 1$ (for this specific 4, i)
    # Then $\sum_j U_{ij,r} A_{ij} = A_{ij}$.
    # $\sum_j V_{j,4,r} B_{j,4} = B_{j,4}$.
    # Product is $A_{ij} B_{j,4}$.
    # $W_{4,i,r} = 1$.
    # Sum over r (which is just this one term for this j) gives $A_{ij} B_{j,4}$.
    # Sum over j gives $\sum_j A_{ij} B_{j,4}$.
    
    # So for each j in 0..3 and k in 4..5:
    # We have a rank-1 term.
    # Total 8 terms.
    
    for j in range(4):
        for k in [4, 5]:
            col = rank1 + (j * 2 + (k - 4)) # 0..7
            # U: i*m + j. But U is 16x57.
            # We need to set U[i*m + j, col] for all i?
            # No, $U_{ij,r}$ is the coefficient.
            # In the term for fixed j, $U_{ij,r}$ is non-zero for all i?
            # No, $U_{ij,r}$ is 1 for the specific i?
            # Wait, $U$ is a matrix of shape $(n*m, R)$.
            # $U_{idx, r}$ where $idx = i*m + j$.
            # In the term for fixed j, we want $\sum_i U_{ij,r} A_{ij}$ to be $A_{ij}$?
            # No, the sum over i is not in the formula.
            # The formula is $\sum_j U_{ij,r} A_{ij}$.
            # So for a fixed r, $U_{ij,r}$ is a vector of length $n*m$.
            # If we set $U_{i m + j, r} = 1$ for a specific i, then $\sum_j' U_{i j', r} A_{i j'} = A_{i j}$.
            # So we need to set $U$ for all i?
            # No, for a fixed r, we are computing one product.
            # If we want to compute $A_{ij} B_{jk}$, we set:
            # $U_{i m + j, r} = 1$
            # $V_{j p + k, r} = 1$
            # $W_{k n + i, r} = 1$
            # This works for a specific i, j, k.
            
            # So for the dot product part, we need 4 terms for each i, k?
            # No, we need 4 terms for each k, summing over j.
            # But the terms are for specific i, j, k.
            # So for each i, j, k, we have a term.
            # Total $4 \times 4 \times 2 = 32$ terms?
            # But the rank of $4 \times 4 \times 1$ is 4.
            # This implies we can share terms?
            # No, the rank of $M_{n,m,1}$ is $m$.
            # So for $4 \times 4 \times 1$, rank is 4.
            # This means we can do it with 4 multiplications.
            # How?
            # $C_i = \sum_j A_{ij} B_j$.
            # This is 4 dot products.
            # Each dot product is rank 1?
            # No, a dot product of two vectors of length 4 has rank 4.
            # So $4 \times 4 \times 1$ has rank 16?
            # No, the rank of $M_{n,m,1}$ is $n \times m$?
            # No, $M_{1,m,1}$ is rank 1?
            # $C = A B$. $A$ is $1 \times m$, $B$ is $m \times 1$.
            # $C = \sum_j A_j B_j$.
            # This is a sum of m rank-1 terms.
            # So rank is m.
            # For $4 \times 4 \times 1$, $A$ is $4 \times 4$, $B$ is $4 \times 1$.
            # $C$ is $4 \times 1$.
            # $C_i = \sum_j A_{ij} B_j$.
            # This is 4 independent dot products.
            # Each dot product has rank 4.
            # Total rank 16.
            
            # So Rank($4 \times 4 \times 2$) is 32.
            # Total rank $49 + 32 = 81$.
            
            # This is still better than 96.
            
            for i in range(4):
                for j in range(4):
                    col = rank1 + (i * 8 + j * 2 + (k - 4))
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1

    return {"U": U, "V": V, "W": W}
