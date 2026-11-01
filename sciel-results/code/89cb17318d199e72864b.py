import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 (Strassen 4x4)
    rank = 56
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block indices:
    # A11: i in [0,1], j in [0,1]
    # A12: i in [0,1], j in [2,3]
    # A21: i in [2,3], j in [0,1]
    # A22: i in [2,3], j in [2,3]
    
    # B11: j in [0,1], k in [0,1]
    # B12: j in [0,1], k in [2,3]
    # B21: j in [2,3], k in [0,1]
    # B22: j in [2,3], k in [2,3]
    
    # C11: i in [0,1], k in [0,1] -> W index k*4+i
    # C12: i in [0,1], k in [2,3] -> W index k*4+i
    # C21: i in [2,3], k in [0,1] -> W index k*4+i
    # C22: i in [2,3], k in [2,3] -> W index k*4+i

    # Define the 7 Strassen blocks
    # Each block l defines X (2x2 linear combos of A) and Y (2x2 linear combos of B)
    # X[r][c] is a list of (i, j, coeff)
    # Y[r][c] is a list of (j, k, coeff)
    
    # Helper to create a list of indices for a 2x2 block of A
    def get_A_block(br, bc):
        # br, bc in {0,1}
        # returns list of (i, j, coeff=1)
        res = []
        for dr in range(2):
            for dc in range(2):
                i = br*2 + dr
                j = bc*2 + dc
                res.append((i, j, 1.0))
        return res

    def get_B_block(br, bc):
        # br, bc in {0,1}
        # returns list of (j, k, coeff=1)
        res = []
        for dr in range(2):
            for dc in range(2):
                j = br*2 + dr
                k = bc*2 + dc
                res.append((j, k, 1.0))
        return res

    def combine_lists(list1, list2, coeff1=1.0, coeff2=1.0):
        # Combines two lists of (idx, coeff) with scaling
        # Note: We assume indices are unique within a block for simplicity, 
        # but here we are summing distinct blocks so indices are disjoint.
        # If indices overlapped, we would need to sum coeffs.
        # For Strassen, the blocks are disjoint.
        res = []
        for idx, c in list1:
            res.append((idx, c * coeff1))
        for idx, c in list2:
            res.append((idx, c * coeff2))
        return res

    # The 7 Strassen formulas
    # l: 0..6
    # X_blocks: list of (block_row, block_col, coeff) for A
    # Y_blocks: list of (block_row, block_col, coeff) for B
    
    strassen_defs = [
        # M1 = (A11 + A22)(B11 + B22)
        {
            'x_blocks': [(0,0,1), (1,1,1)],
            'y_blocks': [(0,0,1), (1,1,1)],
            'c_blocks': [(0,0,1), (1,1,1)] # C11 += M1, C22 += M1
        },
        # M2 = (A21 + A22)B11
        {
            'x_blocks': [(1,0,1), (1,1,1)],
            'y_blocks': [(0,0,1)],
            'c_blocks': [(1,0,1), (1,1,-1)] # C21 += M2, C22 -= M2
        },
        # M3 = A11(B12 - B22)
        {
            'x_blocks': [(0,0,1)],
            'y_blocks': [(0,1,1), (1,1,-1)],
            'c_blocks': [(0,1,1), (1,1,1)] # C12 += M3, C22 += M3
        },
        # M4 = A22(B21 - B11)
        {
            'x_blocks': [(1,1,1)],
            'y_blocks': [(1,0,1), (0,0,-1)],
            'c_blocks': [(0,0,-1), (1,0,1)] # C11 -= M4, C21 += M4
        },
        # M5 = (A11 + A12)B22
        {
            'x_blocks': [(0,0,1), (0,1,1)],
            'y_blocks': [(1,1,1)],
            'c_blocks': [(0,0,1)] # C11 += M5
        },
        # M6 = (A21 - A11)(B11 + B12)
        {
            'x_blocks': [(1,0,1), (0,0,-1)],
            'y_blocks': [(0,0,1), (0,1,1)],
            'c_blocks': [(1,1,1)] # C22 += M6
        },
        # M7 = (A12 - A22)(B21 + B22)
        {
            'x_blocks': [(0,1,1), (1,1,-1)],
            'y_blocks': [(1,0,1), (1,1,1)],
            'c_blocks': [(0,0,-1), (0,1,1)] # C11 -= M7, C12 += M7
        }
    ]

    col = 0
    for l in range(7):
        defn = strassen_defs[l]
        
        # Build X (2x2 matrix of lists) and Y (2x2 matrix of lists)
        # X[r][c] corresponds to the element at row r, col c of the 2x2 matrix X
        # We need to aggregate the linear combinations for each element.
        
        X = [[[] for _ in range(2)] for _ in range(2)]
        Y = [[[] for _ in range(2)] for _ in range(2)]
        
        # Populate X
        for (br, bc, coeff) in defn['x_blocks']:
            # The block (br, bc) contributes to X[br][bc]
            # Wait, the Strassen formulas operate on the 2x2 block matrices.
            # X is the 2x2 matrix of blocks? No.
            # In Strassen 4x4, we view A as a 2x2 matrix of 2x2 blocks.
            # The formula M1 = (A11 + A22)(B11 + B22) means we form a 2x2 matrix X
            # where X[0][0] = A11 + A22? No.
            # The "blocks" A11, etc are 2x2 matrices.
            # The formula is for the multiplication of two 2x2 matrices of blocks.
            # So X is a 2x2 matrix where each entry is a 2x2 matrix (linear combo).
            # But we are decomposing into scalar multiplications.
            # We need to form the specific 2x2 matrix X (of scalars/linear combos) that is being multiplied by Y.
            
            # Let's clarify:
            # We are computing M_l = X_l * Y_l, where X_l and Y_l are 2x2 matrices.
            # The entries of X_l are linear combinations of the elements of A.
            # For M1: X_1 = [[A11+A22, 0], [0, 0]]? No.
            # The Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
            # The operands for these 2x2 multiplications are 2x2 matrices whose entries are 2x2 blocks of A and B.
            # So, for M1, the left operand is a 2x2 matrix L1 where:
            # L1[0][0] = A11 + A22 (this is a 2x2 matrix)
            # L1[0][1] = 0
            # L1[1][0] = 0
            # L1[1][1] = 0
            # This interpretation is WRONG.
            
            # Correct interpretation:
            # Strassen's algorithm for $n=4$ treats the $4 \times 4$ matrices as $2 \times 2$ blocks of $2 \times 2$ matrices.
            # It computes 7 products of $2 \times 2$ matrices.
            # Each product is $M_k = X_k Y_k$.
            # $X_k$ is a $2 \times 2$ matrix where each entry is a $2 \times 2$ matrix (linear combo of A blocks).
            # $Y_k$ is a $2 \times 2$ matrix where each entry is a $2 \times 2$ matrix (linear combo of B blocks).
            # The result $M_k$ is a $2 \times 2$ matrix of $2 \times 2$ matrices.
            # Then we sum these $M_k$ with signs to get the final $C$ blocks.
            
            # So, for each $k$, we need to decompose the product of two $2 \times 2$ matrices ($X_k, Y_k$) into scalar multiplications.
            # But $X_k$ and $Y_k$ are $2 \times 2$ matrices of $2 \times 2$ blocks.
            # This means $X_k$ is effectively a $4 \times 4$ matrix (sparse) and $Y_k$ is a $4 \times 4$ matrix (sparse).
            # The product $M_k$ is a $4 \times 4$ matrix.
            # We decompose $M_k$ into 8 scalar multiplications? No, a general $4 \times 4$ product has rank 16.
            # But $X_k$ and $Y_k$ have a specific structure.
            
            # Actually, the standard Strassen 4x4 rank is 56.
            # This comes from $7 \times 8$.
            # This implies that each of the 7 "block multiplications" is decomposed into 8 scalar multiplications.
            # This is only possible if the "block multiplication" is actually a multiplication of two $2 \times 2$ matrices of SCALARS.
            # This implies that the "blocks" in the Strassen formula for 4x4 are $1 \times 1$? No.
            
            # Let's re-read the standard Strassen 4x4 construction.
            # It uses 7 multiplications of $2 \times 2$ matrices.
            # Each $2 \times 2$ matrix multiplication has rank 8.
            # Total rank 56.
            # So, for each of the 7 terms, we are multiplying two $2 \times 2$ matrices.
            # Let's call them $X^{(k)}$ and $Y^{(k)}$.
            # $X^{(k)}$ is a $2 \times 2$ matrix. Its entries are linear combinations of $A$'s entries.
            # $Y^{(k)}$ is a $2 \times 2$ matrix. Its entries are linear combinations of $B$'s entries.
            # The product $P^{(k)} = X^{(k)} Y^{(k)}$ is a $2 \times 2$ matrix.
            # The entries of $P^{(k)}$ are linear combinations of products of $A$ and $B$ entries.
            # We decompose this $2 \times 2$ product into 8 scalar multiplications.
            # Then we sum these 8 results to form the contribution to the final $C$.
            
            # So, we need to define $X^{(k)}$ and $Y^{(k)}$ as $2 \times 2$ matrices of linear combinations of $A$ and $B$ scalars.
            
            # For M1 = (A11 + A22)(B11 + B22):
            # This is NOT a 2x2 matrix multiplication. It is the product of two 2x2 MATRICES (the blocks).
            # Wait. If we view A as a 2x2 matrix of 2x2 blocks, then A11+A22 is a 2x2 matrix.
            # The product (A11+A22)(B11+B22) is a 2x2 matrix multiplication.
            # So $X^{(1)}$ is a $2 \times 2$ matrix? No.
            # $X^{(1)}$ IS the matrix $A11+A22$. It is a $2 \times 2$ matrix.
            # $Y^{(1)}$ IS the matrix $B11+B22$. It is a $2 \times 2$ matrix.
            # So we are multiplying two $2 \times 2$ matrices.
            # We decompose this $2 \times 2$ product into 8 scalar multiplications.
            
            # So, for each k, we define:
            # X_k: 2x2 matrix where entry (r,c) is a linear combo of A scalars.
            # Y_k: 2x2 matrix where entry (r,c) is a linear combo of B scalars.
            
            # How to get X_k entries?
            # For M1: X_1 = A11 + A22.
            # Entry (0,0) of X_1 is A[0,0] + A[2,2].
            # Entry (0,1) of X_1 is A[0,1] + A[2,3].
            # Entry (1,0) of X_1 is A[1,0] + A[3,2].
            # Entry (1,1) of X_1 is A[1,1] + A[3,3].
            
            # So we can construct X_k and Y_k as 2x2 arrays of lists of (idx, coeff).
            
            X_k = [[[] for _ in range(2)] for _ in range(2)]
            Y_k = [[[] for _ in range(2)] for _ in range(2)]
            
            # Fill X_k
            # We need to map the block definition to the 2x2 entries.
            # The blocks in the definition (e.g., A11) correspond to the whole 2x2 matrix.
            # So if x_blocks contains (0,0,1), it means add A11 to the matrix X_k.
            # A11 contributes to X_k[0][0], X_k[0][1], X_k[1][0], X_k[1][1].
            
            for (br, bc, coeff) in defn['x_blocks']:
                # br, bc is the block index in the 2x2 block matrix.
                # This block is a 2x2 matrix.
                # We add this block to the corresponding position in X_k?
                # No, X_k IS the matrix formed by the sum of these blocks.
                # Wait, if X_k = A11 + A22, then X_k is a 2x2 matrix.
                # The block A11 is the top-left 2x2 submatrix of A.
                # So A11 contributes to all 4 entries of X_k?
                # No. A11 is a 2x2 matrix. A22 is a 2x2 matrix.
                # Their sum is a 2x2 matrix.
                # So yes, the entries of A11 (which are A[0,0]...A[1,1]) add to the entries of X_k (which are X_k[0,0]...X_k[1,1]).
                
                # So for each block (br, bc) with coeff c:
                # The block is located at rows br*2..br*2+1 and cols bc*2..bc*2+1 of A.
                # It contributes to the entries of X_k at the SAME relative positions?
                # No. X_k is a 2x2 matrix. The "block" A11 is a 2x2 matrix.
                # So A11 adds to X_k directly.
                # A22 adds to X_k directly.
                # So X_k[r][c] gets contributions from A[br*2+r, bc*2+c] ?
                # Yes.
                
                for r in range(2):
                    for c in range(2):
                        i = br*2 + r
                        j = bc*2 + c
                        X_k[r][c].append((i, j, coeff))
                        
            for (br, bc, coeff) in defn['y_blocks']:
                for r in range(2):
                    for c in range(2):
                        j = br*2 + r
                        k = bc*2 + c
                        Y_k[r][c].append((j, k, coeff))
            
            # Now X_k and Y_k are 2x2 matrices of linear combos.
            # We decompose the product X_k * Y_k into 8 scalar multiplications.
            # P = X_k Y_k
            # P[0][0] = X_k[0][0]Y_k[0][0] + X_k[0][1]Y_k[1][0]
            # P[0][1] = X_k[0][0]Y_k[0][1] + X_k[0][1]Y_k[1][1]
            # P[1][0] = X_k[1][0]Y_k[0][0] + X_k[1][1]Y_k[1][0]
            # P[1][1] = X_k[1][0]Y_k[0][1] + X_k[1][1]Y_k[1][1]
            
            # The 8 scalar products are:
            # 1. X_k[0][0] * Y_k[0][0] -> P[0][0]
            # 2. X_k[0][1] * Y_k[1][0] -> P[0][0]
            # 3. X_k[0][0] * Y_k[0][1] -> P[0][1]
            # 4. X_k[0][1] * Y_k[1][1] -> P[0][1]
            # 5. X_k[1][0] * Y_k[0][0] -> P[1][0]
            # 6. X_k[1][1] * Y_k[1][0] -> P[1][0]
            # 7. X_k[1][0] * Y_k[0][1] -> P[1][1]
            # 8. X_k[1][1] * Y_k[1][1] -> P[1][1]
            
            scalar_products = [
                ((0,0), (0,0), (0,0)), # X_idx, Y_idx, P_idx
                ((0,1), (1,0), (0,0)),
                ((0,0), (0,1), (0,1)),
                ((0,1), (1,1), (0,1)),
                ((1,0), (0,0), (1,0)),
                ((1,1), (1,0), (1,0)),
                ((1,0), (0,1), (1,1)),
                ((1,1), (1,1), (1,1))
            ]
            
            for sp in scalar_products:
                x_idx, y_idx, p_idx = sp
                x_list = X_k[x_idx[0]][x_idx[1]]
                y_list = Y_k[y_idx[0]][y_idx[1]]
                p_r, p_c = p_idx
                
                # This scalar product contributes to P_k[p_r][p_c]
                # P_k is a 2x2 matrix of 2x2 blocks? No.
                # P_k is the result of multiplying two 2x2 matrices.
                # The entries of P_k are scalars (linear combos of products).
                # But in the context of the 4x4 multiplication, P_k corresponds to a 2x2 block of C?
                # No.
                # Let's trace the indices.
                # X_k and Y_k are 2x2 matrices.
                # Their product P_k is a 2x2 matrix.
                # The entries of P_k are sums of products of A and B elements.
                # These entries correspond to specific 2x2 blocks of C?
                # No, the entries of P_k are SCALARS?
                # No, X_k[0][0] is a linear combo of A scalars.
                # So P_k[0][0] is a linear combo of products of A and B scalars.
                # This corresponds to ONE element of the final C matrix?
                # No.
                
                # Let's re-evaluate the block structure.
                # A is 4x4. A11 is 2x2.
                # X_k = A11 + A22. X_k is a 2x2 matrix.
                # X_k[0][0] = A[0,0] + A[2,2]. This is a scalar.
                # So X_k is a 2x2 matrix of scalars.
                # Y_k is a 2x2 matrix of scalars.
                # P_k = X_k Y_k is a 2x2 matrix of scalars.
                # P_k[0][0] is a scalar (sum of products).
                # This scalar corresponds to which element of C?
                
                # The Strassen formula says:
                # C11 (2x2 block) = M1 - M4 + M5 - M7.
                # M1 is the result of (A11+A22)(B11+B22).
                # M1 is a 2x2 matrix.
                # So M1[0][0] contributes to C[0][0].
                # M1[0][1] contributes to C[0][1].
                # M1[1][0] contributes to C[1][0].
                # M1[1][1] contributes to C[1][1].
                
                # So, P_k[p_r][p_c] contributes to the block C_block corresponding to (p_r, p_c) of the 2x2 block matrix.
                # The block (p_r, p_c) of C is a 2x2 matrix.
                # Wait. P_k is a 2x2 matrix.
                # So P_k[0][0] is a scalar.
                # But C11 is a 2x2 matrix.
                # How does a scalar P_k[0][0] contribute to a 2x2 matrix C11?
                
                # Ah, I see the confusion.
                # In the standard Strassen 4x4, the "blocks" A11 are 2x2 matrices.
                # The multiplication M1 = (A11+A22)(B11+B22) is a multiplication of two 2x2 MATRICES.
                # The result M1 is a 2x2 MATRIX.
                # So M1 has 4 entries: M1[0][0], M1[0][1], M1[1][0], M1[1][1].
                # Each of these entries is a scalar (sum of products of A and B elements).
                # And these 4 scalars form the 2x2 block C11?
                # No, C11 is a 2x2 block of C.
                # So M1[0][0] is C[0][0].
                # M1[0][1] is C[0][1].
                # M1[1][0] is C[1][0].
                # M1[1][1] is C[1][1].
                
                # So, for each scalar product in the decomposition of X_k Y_k, we identify which entry P_k[p_r][p_c] it contributes to.
                # Then we map that entry to the specific element C[i][k].
                # The block index (p_r, p_c) determines the 2x2 sub-block of C.
                # Within that sub-block, the entry is at (0,0) or (0,1) etc?
                # No. P_k is the 2x2 matrix M_l.
                # So P_k[p_r][p_c] is the element at row p_r, col p_c of the 2x2 matrix M_l.
                # This element corresponds to C[br*2 + p_r, bc*2 + p_c]?
                # No. The block C_block is determined by the Strassen output formula.
                # e.g. M1 contributes to C11 and C22.
                # M1 is a 2x2 matrix.
                # M1[0][0] -> C11[0][0] -> C[0][0].
                # M1[0][1] -> C11[0][1] -> C[0][1].
                # M1[1][0] -> C11[1][0] -> C[1][0].
                # M1[1][1] -> C11[1][1] -> C[1][1].
                
                # So, for a given M_l, we know which C blocks it contributes to, and with what sign.
                # For each entry (r,c) of M_l (which is P_k[r][c]), we add it to the corresponding entry of the target C block.
                
                # So, for this scalar product contributing to P_k[p_r][p_c]:
                # We look at defn['c_blocks'].
                # Each entry is (c_br, c_bc, coeff).
                # This means P_k[p_r][p_c] contributes coeff * P_k[p_r][p_c] to the C block (c_br, c_bc).
                # The specific element in C is at:
                # i = c_br * 2 + p_r
                # k = c_bc * 2 + p_c
                
                # Now we can fill U, V, W.
                
                if col >= rank:
                    continue
                    
                # Fill U
                for (i_x, j_x, c_x) in x_list:
                    U[i_x*m + j_x, col] += c_x
                # Fill V
                for (j_y, k_y, c_y) in y_list:
                    V[j_y*p + k_y, col] += c_y
                # Fill W
                for (c_br, c_bc, c_coeff) in defn['c_blocks']:
                    i = c_br * 2 + p_r
                    k = c_bc * 2 + p_c
                    W[k*n + i, col] += c_coeff
                    
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
