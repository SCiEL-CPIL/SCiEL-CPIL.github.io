"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Strategy:
    # 1. Try Rank 161: Strassen (2x2 blocks) + Laderman (3x3 blocks)
    #    6x6 matrix viewed as 2x2 blocks of 3x3 matrices.
    #    Block mult: Strassen (7 terms).
    #    Inner mult: Laderman (23 terms).
    #    Total Rank: 7 * 23 = 161.
    
    # 2. Fallback Rank 189: Strassen (2x2 blocks) + Schoolbook (3x3 blocks)
    #    OR Schoolbook (3x3 blocks) + Strassen (2x2 blocks).
    #    Current program uses 3x3 blocks of 2x2 (27*7=189).
    
    # Helper: Verify decomposition
    def verify_decomp(U, V, W, n, m, p):
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Laderman 3x3 algorithm (23 terms) ---
    # Each term is (u, v, w) where u, v, w are 3x3 matrices (flat, row-major)
    # The tensor identity: sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = delta_{i,k} (for output C[i,k] = sum_j A[i,j]B[j,k])
    # Wait, the convention is T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    # For 3x3 inner: U[i,j,t] * V[j,k,t] * W[k,i,t] should give 1 when i==k (output index) and 0 otherwise?
    # No. The matrix multiplication tensor M_{n} has M[i,j,k] = 1 if i==k (where i is row of A, j is col of A/row of B, k is col of B).
    # Actually standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T_{i,j,k} = delta_{i,k} ... no.
    # T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # In the decomposition: C_{i,k} = sum_t (sum_j U_{i,j,t} A_{i,j}) (sum_j V_{j,k,t} B_{j,k}) W_{k,i,t}?
    # Let's check the contract:
    # U shape (n*m, rank) -> U[i*m+j, t]
    # V shape (m*p, rank) -> V[j*p+k, t]
    # W shape (p*n, rank) -> W[k*n+i, t]
    # Product: sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    # This must equal 1 for all i,j,k? No.
    # The tensor for matrix mult is T[i,j,k] = 1.
    # Wait, the tensor T such that T(A,B) = AB is defined by T_{i,j,k} = delta_{i,k}? No.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in the output C_{i,k}.
    # The output is indexed by (i,k). The input indices are (i,j) and (j,k).
    # So the tensor is a 3-tensor T[i,j,k] where i is row of A, j is shared index, k is col of B.
    # T[i,j,k] = 1.
    # So we need sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 for all i,j,k.
    # Note: W is indexed by (k,i) in the product term W[k*n+i, t].
    # So W[k,i,t] multiplies the term.
    
    # Laderman's algorithm for 3x3:
    # C = A * B
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    # 23 terms. I will use a known correct set.
    # Reference: "A 23-term algorithm for multiplying 3x3 matrices"
    # The terms are often listed as (x, y, z) such that sum x_t * y_t * z_t = C.
    # Here x corresponds to U, y to V, z to W.
    # U[i,j,t] is coeff of A[i,j]
    # V[j,k,t] is coeff of B[j,k]
    # W[k,i,t] is coeff for output C[i,k] (note indices k,i in W)
    
    # Let's use the terms from a reliable source.
    # I will hardcode the 23 terms.
    # If I get them wrong, verification will fail and I return 189.
    
    # Terms (u, v, w) as 3x3 flat arrays.
    # u: 3x3, v: 3x3, w: 3x3.
    # Note: W in the formula is W[k*n+i, t], which corresponds to W_mat[k, i] in my local 3x3 notation.
    # So the third element of the tuple is the matrix Z where Z[k,i] is the coefficient.
    
    laderman_terms = [
        # 1
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        # 2
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # 3
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        # 4
        ([1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # 5
        ([0,0,0, 0,1,0, 1,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # 6
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,1,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # 7
        ([1,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # 8
        ([0,0,0, 0,0,0, 0,1,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # 9
        ([0,1,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 10
        ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 11
        ([0,0,1, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 12
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # 13
        ([1,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 14
        ([0,1,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 15
        ([0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 16
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 17
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 18
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 19
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 20
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 21
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 22
        ([1,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 23
        ([0,0,0, 0,0,1, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Verify Laderman locally before using it
    def verify_laderman(terms):
        # Check sum_t u_t[i,j] v_t[j,k] w_t[k,i] = 1 for all i,j,k
        # Wait, the tensor is T[i,j,k] = 1.
        # My previous check: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # U[i,j] is u_t[i,j]
        # V[j,k] is v_t[j,k]
        # W[k,i] is w_t[k,i]
        # So sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] should be 1.
        
        u_np = np.array([t[0] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        v_np = np.array([t[1] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        w_np = np.array([t[2] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        
        # u: (t, i, j)
        # v: (t, j, k)
        # w: (t, k, i)
        
        # T[i,j,k] = sum_t u[t,i,j] v[t,j,k] w[t,k,i]
        # einsum: 'tij,tjk,tki->ijk'
        T = np.einsum('tij,tjk,tki->ijk', u_np, v_np, w_np)
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    laderman_ok = verify_laderman(laderman_terms)
    
    if laderman_ok:
        # Construct 6x6 decomposition using Strassen(2x2) of Laderman(3x3)
        # Block size 3, 2x2 blocks
        block_size = 3
        blocks_per_dim = 2
        
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        rank = 7 * 23
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
                        for (pi, pj), val in u_coeffs.items():
                            for (si, sj), lval in enumerate(u_coeffs.items()): # dummy
                                pass
                            
                            # Actually, u_coeffs is a dict of block offsets.
                            # For each block offset, we add the Laderman terms scaled by val.
                            
                        # Let's restructure
                        pass
                        
        # Restructure construction
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
                        # u_coeffs: dict {(block_row, block_col): val}
                        # We need to iterate over the non-zero entries in u_coeffs
                        # But u_coeffs is a dict.
                        for (bi, bj), u_val in u_coeffs.items():
                            # For this Strassen term, the U part has value u_val at block (bi, bj)
                            # We need to add u_val * Laderman_U[t] to the appropriate position
                            # Laderman_U[t] is a 3x3 matrix
                            for t in range(23):
                                u_mat = np.array(laderman_terms[t][0]).reshape(3,3)
                                v_mat = np.array(laderman_terms[t][1]).reshape(3,3)
                                w_mat = np.array(laderman_terms[t][2]).reshape(3,3)
                                
                                # U contribution:
                                # Global row: I*3 + si, Global col: J*3 + sj
                                # U[global_row*m + global_col, col_idx] += u_val * u_mat[si, sj]
                                # But wait, u_coeffs only tells us WHICH block has a non-zero coefficient.
                                # The coefficient is u_val.
                                # The Laderman matrix u_mat provides the internal structure.
                                # So for each (si, sj) in 0..2:
                                #   U[I*3+si, J*3+sj] += u_val * u_mat[si, sj]
                                
                                for si in range(3):
                                    for sj in range(3):
                                        gr = I * 3 + si
                                        gc = J * 3 + sj
                                        U[gr * m + gc][col_idx] += u_val * u_mat[si, sj]
                                        
                                # V contribution:
                                # v_coeffs: {(block_row, block_col): val}
                                # But we are inside the loop for a specific Strassen term.
                                # The Strassen term defines u_coeffs, v_coeffs, w_coeffs.
                                # So for the same t, we add v_val * v_mat[si, sj] to V at block (bk, bl) ?
                                # No, v_coeffs is a dict of block offsets for V.
                                # So we must also iterate over v_coeffs.
                                
                                # This nested structure is tricky.
                                # Let's flatten the loops.
                                
                                # Actually, the Strassen term is a combination.
                                # U_part = sum_{(bi,bj) in u_coeffs} u_val(bi,bj) * (Block_{bi,bj} of A)
                                # In the tensor decomposition, the term t in the outer product is:
                                # U_{i,j,t} = sum_{(bi,bj)} u_val(bi,bj) * u_mat[si,sj] where i=bi*3+si, j=bj*3+sj
                                # This is what I did above for U.
                                
                                # Now for V.
                                # V_{j,k,t} = sum_{(bk,bl)} v_val(bk,bl) * v_mat[sk,sl] where j=bk*3+sk, k=bl*3+sl
                                # And W.
                                # W_{k,i,t} = sum_{(bk,bi)} w_val(bk,bi) * w_mat[sk,si] where k=bk*3+sk, i=bi*3+si
                                
                                # So I need to add contributions from v_coeffs and w_coeffs as well.
                                
                                # Let's restart the inner loop for clarity.
                                pass
                        pass
                    pass
                pass
            pass
        pass
        
        # Correct Construction
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
                        # For each Laderman term t
                        for t in range(23):
                            u_mat = np.array(laderman_terms[t][0]).reshape(3,3)
                            v_mat = np.array(laderman_terms[t][1]).reshape(3,3)
                            w_mat = np.array(laderman_terms[t][2]).reshape(3,3)
                            
                            # Add U contributions
                            for (bi, bj), u_val in u_coeffs.items():
                                for si in range(3):
                                    for sj in range(3):
                                        gr = I * 3 + si
                                        gc = J * 3 + sj
                                        # Wait, u_coeffs keys are (bi, bj) relative to the block grid?
                                        # No, u_coeffs is the Strassen pattern.
                                        # The Strassen pattern applies to the 2x2 block grid.
                                        # So (bi, bj) are indices in 0..1.
                                        # But I, J are the current block indices in the 6x6 matrix?
                                        # No, I, J, K are the block indices in the 2x2 grid.
                                        # The Strassen algorithm computes C_{I,K} = sum_J ...
                                        # The decomposition is for the whole 6x6 matrix.
                                        # The blocks are A_{I,J}, B_{J,K}, C_{I,K}.
                                        # Strassen computes C_{I,K} using combinations of A_{*,*} and B_{*,*}.
                                        # So for a fixed (I, J, K) in the outer loops?
                                        # No, Strassen is applied to the 2x2 block matrix.
                                        # The output C is 2x2 blocks.
                                        # C_{I,K} is computed.
                                        # The terms in Strassen involve A_{bi,bj} and B_{bk,bl}.
                                        # So for a specific Strassen term, it contributes to C_{I,K} (where I,K are the output block indices).
                                        # But the Strassen term itself is defined by the patterns u_coeffs, v_coeffs, w_coeffs.
                                        # u_coeffs tells which A blocks are involved.
                                        # v_coeffs tells which B blocks are involved.
                                        # w_coeffs tells which C blocks are involved.
                                        
                                        # So, for a given Strassen term (u_coeffs, v_coeffs, w_coeffs):
                                        # It contributes to the tensor T.
                                        # The U part: For each (bi, bj) in u_coeffs with val u_val:
                                        #   The block A_{bi, bj} is used.
                                        #   The internal structure is u_mat.
                                        #   So U[bi*3+si, bj*3+sj, t] += u_val * u_mat[si, sj]
                                        
                                        # Similarly for V and W.
                                        
                                        # So I should NOT loop over I, J, K in the outer loops if I am iterating over Strassen terms that cover the whole 2x2 grid.
                                        # The Strassen terms ARE the decomposition of the 2x2 block multiplication.
                                        # So I only loop over the 7 Strassen terms.
                                        
                                        # Let's restart the construction logic completely.
                                        pass
                            pass
                        pass
                    pass
                pass
            pass
        pass
        
        # Final Correct Logic
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
            for t in range(23):
                u_mat = np.array(laderman_terms[t][0]).reshape(3,3)
                v_mat = np.array(laderman_terms[t][1]).reshape(3,3)
                w_mat = np.array(laderman_terms[t][2]).reshape(3,3)
                
                # U
                for (bi, bj), u_val in u_coeffs.items():
                    for si in range(3):
                        for sj in range(3):
                            gr = bi * 3 + si
                            gc = bj * 3 + sj
                            U[gr * m + gc][col_idx] += u_val * u_mat[si, sj]
                            
                # V
                for (bk, bl), v_val in v_coeffs.items():
                    for sk in range(3):
                        for sl in range(3):
                            gr = bk * 3 + sk
                            gc = bl * 3 + sl
                            V[gr * p + gc][col_idx] += v_val * v_mat[sk, sl]
                            
                # W
                # w_coeffs keys are (bi, bk)?
                # In Strassen, W maps to C blocks.
                # C_{bi, bk}? No, C is (I, K).
                # The Strassen pattern for W usually has keys (bi, bk) corresponding to C_{bi, bk}.
                # But the contract says W[k*n+i, t].
                # So W is indexed by (k, i).
                # In the Strassen pattern, the W matrix is 2x2.
                # Let's assume w_coeffs keys are (i_block, k_block) or (k_block, i_block)?
                # Standard Strassen:
                # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
                # So W has 1 at (0,0) and (1,1).
                # If w_coeffs is {(0,0): 1, (1,1): 1}, then keys are (i_block, k_block)?
                # If so, W[i_block*3+si, k_block*3+sk] += val * w_mat[si, sk]?
                # But W is stored as W[k*n+i, t].
                # So W[row, col] in the flat array is W[k*n+i].
                # So row index in flat array is k*n + i.
                # My W_mat is 3x3.
                # If w_coeffs key is (i_block, k_block), then:
                # i = i_block*3 + si
                # k = k_block*3 + sk
                # W_flat_index = k*n + i
                # W[W_flat_index][col_idx] += val * w_mat[si, sk]?
                # Wait, w_mat[si, sk] corresponds to W[k,i] or W[i,k]?
                # In the tensor product, W[k,i,t] multiplies U[i,j,t] V[j,k,t].
                # So w_mat should be the matrix Z where Z[k,i] is the coeff.
                # In my Laderman terms, the third element is w_t.
                # I defined w_t as a 3x3 matrix.
                # In verify_laderman, I used w[t,k,i].
                # So w_mat[sk, si] is the coefficient for k=sk, i=si.
                # So if w_coeffs key is (i_block, k_block), then:
                # i = i_block*3 + si
                # k = k_block*3 + sk
                # Coeff is w_mat[sk, si]
                # W_flat_index = k*n + i
                # W[W_flat_index][col_idx] += val * w_mat[sk, si]
                
                for (bi, bk), w_val in w_coeffs.items():
                    # Assume keys are (i_block, k_block)
                    for si in range(3):
                        for sk in range(3):
                            i = bi * 3 + si
                            k = bk * 3 + sk
                            W_flat_idx = k * n + i
                            W[W_flat_idx][col_idx] += w_val * w_mat[sk, si]
                            
                col_idx += 1
        
        # Verify
        if verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Rank 189
    block_size = 2
    blocks_per_dim = 3
    
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
