"""Matrix multiplication 4,4,5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0

    n, m, p = 4, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    def generate_strassen(N):
        """
        Generates the Strassen decomposition for NxN x NxN multiplication.
        Returns a list of (U, V, W) where U, V, W are lists of length N*N.
        """
        if N == 1:
            return [([1.0], [1.0], [1.0])]
        
        N2 = N // 2
        sub_decomp = generate_strassen(N2)
        
        # Helper to create a N*N matrix from a N2*N2 block and position
        # Positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        def expand_block(block, pos, coeff=1.0):
            """
            Expands a N2*N2 block into a N*N matrix at position pos.
            block is a list of length N2*N2.
            pos is 0, 1, 2, or 3.
            """
            mat = [0.0] * (N * N)
            r_offset = (pos // 2) * N2
            c_offset = (pos % 2) * N2
            for i in range(N2):
                for j in range(N2):
                    mat[(r_offset + i) * N + (c_offset + j)] = block[i * N2 + j] * coeff
            return mat

        def combine_blocks(blocks_coeffs):
            """
            Combines multiple (block, pos, coeff) tuples into a single N*N matrix.
            blocks_coeffs: list of (block, pos, coeff)
            """
            mat = [0.0] * (N * N)
            for block, pos, coeff in blocks_coeffs:
                expanded = expand_block(block, pos, coeff)
                for i in range(N * N):
                    mat[i] += expanded[i]
            return mat

        terms = []
        
        # Define the 7 Strassen terms
        # Each term is a tuple (U_blocks, V_blocks, W_blocks)
        # U_blocks: list of (sub_U, pos, coeff)
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22 (coeff +1)
        # Note: The sub-decomp is for the product of the combined matrices.
        # So for each term in sub_decomp, we generate 1 term for M1.
        
        for sub_U, sub_V, sub_W in sub_decomp:
            
            # M1
            U1 = combine_blocks([(sub_U, 0, 1.0), (sub_U, 3, 1.0)])
            V1 = combine_blocks([(sub_V, 0, 1.0), (sub_V, 3, 1.0)])
            W1 = combine_blocks([(sub_W, 0, 1.0), (sub_W, 3, 1.0)])
            terms.append((U1, V1, W1))
            
            # M2 = (A21+A22)B11
            # U: A21, A22
            # V: B11
            # W: C21 (+1), C22 (-1)
            U2 = combine_blocks([(sub_U, 2, 1.0), (sub_U, 3, 1.0)])
            V2 = combine_blocks([(sub_V, 0, 1.0)])
            W2 = combine_blocks([(sub_W, 2, 1.0), (sub_W, 3, -1.0)])
            terms.append((U2, V2, W2))
            
            # M3 = A11(B12-B22)
            # U: A11
            # V: B12 (+1), B22 (-1)
            # W: C12 (+1), C22 (+1)
            U3 = combine_blocks([(sub_U, 0, 1.0)])
            V3 = combine_blocks([(sub_V, 1, 1.0), (sub_V, 3, -1.0)])
            W3 = combine_blocks([(sub_W, 1, 1.0), (sub_W, 3, 1.0)])
            terms.append((U3, V3, W3))
            
            # M4 = A22(B21-B11)
            # U: A22
            # V: B21 (+1), B11 (-1)
            # W: C11 (+1), C21 (+1)
            U4 = combine_blocks([(sub_U, 3, 1.0)])
            V4 = combine_blocks([(sub_V, 2, 1.0), (sub_V, 0, -1.0)])
            W4 = combine_blocks([(sub_W, 0, 1.0), (sub_W, 2, 1.0)])
            terms.append((U4, V4, W4))
            
            # M5 = (A11+A12)B22
            # U: A11, A12
            # V: B22
            # W: C11 (-1), C12 (+1)
            U5 = combine_blocks([(sub_U, 0, 1.0), (sub_U, 1, 1.0)])
            V5 = combine_blocks([(sub_V, 3, 1.0)])
            W5 = combine_blocks([(sub_W, 0, -1.0), (sub_W, 1, 1.0)])
            terms.append((U5, V5, W5))
            
            # M6 = (A21-A11)(B11+B12)
            # U: A21 (+1), A11 (-1)
            # V: B11, B12
            # W: C22 (+1)
            U6 = combine_blocks([(sub_U, 2, 1.0), (sub_U, 0, -1.0)])
            V6 = combine_blocks([(sub_V, 0, 1.0), (sub_V, 1, 1.0)])
            W6 = combine_blocks([(sub_W, 3, 1.0)])
            terms.append((U6, V6, W6))
            
            # M7 = (A12-A22)(B21+B22)
            # U: A12 (+1), A22 (-1)
            # V: B21, B22
            # W: C11 (+1)
            U7 = combine_blocks([(sub_U, 1, 1.0), (sub_U, 3, -1.0)])
            V7 = combine_blocks([(sub_V, 2, 1.0), (sub_V, 3, 1.0)])
            W7 = combine_blocks([(sub_W, 0, 1.0)])
            terms.append((U7, V7, W7))
            
        return terms

    # Generate the 49 terms for 4x4x4
    strassen_terms = generate_strassen(4)
    
    # Prepare the final matrices
    # U: 16 x rank
    # V: 20 x rank
    # W: 20 x rank
    
    total_rank = len(strassen_terms) + 16  # 49 + 16 = 65
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Fill in Strassen terms (columns 0 to 48)
    for idx, (u_list, v_list, w_list) in enumerate(strassen_terms):
        # u_list, v_list, w_list are length 16
        # Map to U[i*4+j, idx]
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, idx] = u_list[i * 4 + j]
                
        # Map to V[j*5+k, idx] for k in 0..3
        for i in range(4): # This is j in the tensor
            for k in range(4):
                V[i * 5 + k, idx] = v_list[i * 4 + k]
                
        # Map to W[k*4+i, idx] for k in 0..3
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, idx] = w_list[k * 4 + i]
                
    # Fill in Schoolbook terms for the 5th column (columns 49 to 64)
    for i in range(4):
        for j in range(4):
            c_idx = 49 + i * 4 + j
            U[i * 4 + j, c_idx] = 1.0
            V[j * 5 + 4, c_idx] = 1.0
            W[4 * 4 + i, c_idx] = 1.0

    # Verification (optional, but good for debugging if it fails)
    # tensor = np.zeros((4, 4, 5))
    # for c in range(total_rank):
    #     for i in range(4):
    #         for j in range(4):
    #             for k in range(5):
    #                 tensor[i, j, k] += U[i*4+j, c] * V[j*5+k, c] * W[k*4+i, c]
    # if not np.allclose(tensor, 1.0):
    #     # Fallback to schoolbook if construction failed
    #     pass

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
