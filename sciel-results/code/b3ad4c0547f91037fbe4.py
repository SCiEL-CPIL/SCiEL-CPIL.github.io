import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by splitting the p dimension (7) into 2, 2, 2, 1.
    # For each p_block of size 2, we use a rank 11 decomposition of M_{2,3,2}.
    # For the p_block of size 1, we use a rank 6 decomposition of M_{2,3,1}.
    
    # Initialize U, V, W as lists of lists (or numpy arrays)
    # Total rank = 3 * 11 + 6 = 39
    total_rank = 39
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    rank_offset = 0
    
    # Helper to add a decomposition block
    # block_p_start: index in p dimension where this block starts
    # rank_block: number of ranks used by this block
    # u_block, v_block, w_block: matrices of shapes (n*m, rank_block), (m*block_p_size, rank_block), (block_p_size*n, rank_block)
    def add_block(block_p_start, block_p_size, u_block, v_block, w_block):
        nonlocal rank_offset
        U[:, rank_offset:rank_offset+block_p_size*0+u_block.shape[1]] = u_block # Wait, shapes must match
        # Actually, V is (m*p, rank). The block corresponds to rows [block_p_start : block_p_start + block_p_size] in the p dimension of B?
        # No, B is (m, p). Index is j*p + k.
        # So for a block of p of size s starting at p_start, the rows in V are j*p + (p_start + k).
        # This is not a contiguous block in V if p is the last index?
        # V index: j*p + k.
        # If we fix k range [p_start, p_start+s), the indices are j*p + k.
        # This is NOT a simple slice of V.
        # We must fill V carefully.
        
        pass

    # Let's do it manually for clarity and correctness.
    
    # Block 1: p in [0, 2)
    # M_{2,3,2} decomposition
    # A part: A[:, 0:2] (rows 0,1 of A? No, A is 2x3. Columns 0,1 of A)
    # B part: B[0:2, 0:2] (Rows 0,1 of B, Cols 0,1 of B)
    # Wait, A is 2x3. B is 3x2.
    # A[:, 0:2] is 2x2. B[0:2, :] is 2x2.
    # Strassen on A[:, 0:2] and B[0:2, :].
    
    # Strassen for M_{2,2,2}
    # A_s = A[:, 0:2], B_s = B[0:2, :]
    # U_s, V_s, W_s for M_{2,2,2}
    # U_s shape (4, 7), V_s shape (4, 7), W_s shape (4, 7)
    # U_s[i*2+j, k] corresponds to A[i, j]
    # V_s[j*2+k, k] corresponds to B[j, k]
    # W_s[k*2+i, k] corresponds to C[i, k]
    
    # Let's define the Strassen decomposition explicitly.
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-a10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a11)(b01+b10)
    # M7 = (a01-a10)(b00+b10)
    
    # We will create a function to add a Strassen block.
    
    def add_strassen_block(p_start, a_col_start, b_row_start):
        # a_col_start: column in A (0..2)
        # b_row_start: row in B (0..2)
        # p_start: column in B (0..6)
        # This block computes C[:, p_start:p_start+2] += A[:, a_col_start:a_col_start+2] @ B[b_row_start:b_row_start+2, p_start:p_start+2]
        
        # U indices for A part:
        # A is 2x3. We use columns a_col_start, a_col_start+1.
        # Rows 0, 1.
        # U row index: i*3 + j.
        # i in 0..1, j in a_col_start..a_col_start+1.
        
        # V indices for B part:
        # B is 3x7. We use rows b_row_start, b_row_start+1.
        # Cols p_start, p_start+1.
        # V row index: j*7 + k.
        # j in b_row_start..b_row_start+1, k in p_start..p_start+1.
        
        # W indices for C part:
        # C is 2x7. We use cols p_start, p_start+1.
        # Rows 0, 1.
        # W row index: k*2 + i.
        # k in p_start..p_start+1, i in 0..1.
        
        # Strassen multipliers:
        # We need to map the 7 multiplications to the global U, V, W.
        
        # Let's define the local U, V, W for Strassen.
        # Local U (4x7):
        # M1: a00, a11 -> U[0]=1, U[3]=1
        # M2: a10, a11 -> U[2]=1, U[3]=1
        # M3: a00 -> U[0]=1
        # M4: a11, a10 -> U[3]=1, U[2]=-1
        # M5: a00, a01 -> U[0]=1, U[1]=1
        # M6: a10, a11 -> U[2]=1, U[3]=-1
        # M7: a01, a10 -> U[1]=1, U[2]=-1
        
        # Local V (4x7):
        # M1: b00, b11 -> V[0]=1, V[3]=1
        # M2: b00 -> V[0]=1
        # M3: b01, b11 -> V[1]=1, V[3]=-1
        # M4: b10 -> V[2]=1
        # M5: b11 -> V[3]=1
        # M6: b01, b10 -> V[1]=1, V[2]=1
        # M7: b00, b10 -> V[0]=1, V[2]=1
        
        # Local W (4x7):
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add these to the global arrays.
        
        for mult_idx in range(7):
            global_rank = rank_offset + mult_idx
            
            # U
            if mult_idx == 0: # M1
                U[a_col_start * 0 + 0, global_rank] += 1 # i=0, j=a_col_start -> 0*3 + a_col_start
                U[1 * 3 + a_col_start + 1, global_rank] += 1 # i=1, j=a_col_start+1
            elif mult_idx == 1: # M2
                U[1 * 3 + a_col_start, global_rank] += 1
                U[1 * 3 + a_col_start + 1, global_rank] += 1
            elif mult_idx == 2: # M3
                U[0 * 3 + a_col_start, global_rank] += 1
            elif mult_idx == 3: # M4
                U[1 * 3 + a_col_start + 1, global_rank] += 1
                U[1 * 3 + a_col_start, global_rank] += -1
            elif mult_idx == 4: # M5
                U[0 * 3 + a_col_start, global_rank] += 1
                U[0 * 3 + a_col_start + 1, global_rank] += 1
            elif mult_idx == 5: # M6
                U[1 * 3 + a_col_start, global_rank] += 1
                U[1 * 3 + a_col_start + 1, global_rank] += -1
            elif mult_idx == 6: # M7
                U[0 * 3 + a_col_start + 1, global_rank] += 1
                U[1 * 3 + a_col_start, global_rank] += -1
            
            # V
            if mult_idx == 0: # M1
                V[b_row_start * 7 + p_start, global_rank] += 1
                V[(b_row_start + 1) * 7 + p_start + 1, global_rank] += 1
            elif mult_idx == 1: # M2
                V[b_row_start * 7 + p_start, global_rank] += 1
            elif mult_idx == 2: # M3
                V[b_row_start * 7 + p_start + 1, global_rank] += 1
                V[(b_row_start + 1) * 7 + p_start + 1, global_rank] += -1
            elif mult_idx == 3: # M4
                V[(b_row_start + 1) * 7 + p_start, global_rank] += 1
            elif mult_idx == 4: # M5
                V[(b_row_start + 1) * 7 + p_start + 1, global_rank] += 1
            elif mult_idx == 5: # M6
                V[b_row_start * 7 + p_start + 1, global_rank] += 1
                V[(b_row_start + 1) * 7 + p_start, global_rank] += 1
            elif mult_idx == 6: # M7
                V[b_row_start * 7 + p_start, global_rank] += 1
                V[(b_row_start + 1) * 7 + p_start, global_rank] += 1
            
            # W
            # C indices: k*2 + i
            # k is p_start, p_start+1
            # i is 0, 1
            
            if mult_idx == 0: # M1
                W[p_start * 2 + 0, global_rank] += 1
                W[(p_start + 1) * 2 + 1, global_rank] += 1
            elif mult_idx == 1: # M2
                W[p_start * 2 + 1, global_rank] += -1
                W[(p_start + 1) * 2 + 0, global_rank] += 1
            elif mult_idx == 2: # M3
                W[p_start * 2 + 1, global_rank] += 1
                W[(p_start + 1) * 2 + 1, global_rank] += 1
            elif mult_idx == 3: # M4
                W[p_start * 2 + 0, global_rank] += 1
                W[(p_start + 1) * 2 + 0, global_rank] += 1
            elif mult_idx == 4: # M5
                W[p_start * 2 + 0, global_rank] += -1
                W[p_start * 2 + 1, global_rank] += 1
            elif mult_idx == 5: # M6
                W[(p_start + 1) * 2 + 1, global_rank] += 1
            elif mult_idx == 6: # M7
                W[p_start * 2 + 0, global_rank] += 1

        rank_offset += 7

    # Block 2: M_{2,1,2} for A[:, 2] and B[2, 0:2]
    # A part: column 2.
    # B part: row 2, cols 0,1.
    # C part: cols 0,1.
    
    def add_vector_block(p_start, a_col, b_row):
        # A is 2x1 (col a_col). B is 1x2 (row b_row, cols p_start, p_start+1).
        # 4 multiplications.
        # M1: A[0, a_col] * B[b_row, p_start]
        # M2: A[0, a_col] * B[b_row, p_start+1]
        # M3: A[1, a_col] * B[b_row, p_start]
        # M4: A[1, a_col] * B[b_row, p_start+1]
        
        for mult_idx in range(4):
            global_rank = rank_offset + mult_idx
            
            # U
            if mult_idx == 0 or mult_idx == 1:
                U[0 * 3 + a_col, global_rank] += 1
            else:
                U[1 * 3 + a_col, global_rank] += 1
            
            # V
            if mult_idx == 0 or mult_idx == 2:
                V[b_row * 7 + p_start, global_rank] += 1
            else:
                V[b_row * 7 + p_start + 1, global_rank] += 1
            
            # W
            if mult_idx == 0:
                W[p_start * 2 + 0, global_rank] += 1
            elif mult_idx == 1:
                W[p_start * 2 + 1, global_rank] += 1
            elif mult_idx == 2:
                W[p_start * 2 + 0, global_rank] += 1 # Wait, C[1, p_start]
                W[p_start * 2 + 0, global_rank] -= 1 # No, C[1, p_start] is index (p_start)*2 + 1
                # Let's re-verify W index: k*n + i. k=p_start, i=1. Index = p_start*2 + 1.
                W[p_start * 2 + 1, global_rank] += 1
                W[p_start * 2 + 0, global_rank] -= 1 # Remove previous error
            elif mult_idx == 3:
                W[(p_start + 1) * 2 + 1, global_rank] += 1

        rank_offset += 4

    # We need to handle the 3 blocks of size 2 and 1 block of size 1.
    # The p dimension is 7.
    # Blocks:
    # 1. p in [0, 2): A[:, 0:2], B[0:2, 0:2] (Strassen)
    # 2. p in [0, 2): A[:, 2], B[2, 0:2] (Vector)
    # 3. p in [2, 4): A[:, 0:2], B[0:2, 2:4] (Strassen)
    # 4. p in [2, 4): A[:, 2], B[2, 2:4] (Vector)
    # 5. p in [4, 6): A[:, 0:2], B[0:2, 4:6] (Strassen)
    # 6. p in [4, 6): A[:, 2], B[2, 4:6] (Vector)
    # 7. p in [6, 7): A[:, 0:3], B[0:3, 6:7] (Schoolbook 6)
    
    # Wait, the Strassen block uses A[:, 0:2] and B[0:2, :].
    # The Vector block uses A[:, 2] and B[2, :].
    # So for each p-block of size 2, we need both.
    
    add_strassen_block(0, 0, 0)
    add_vector_block(0, 2, 2)
    
    add_strassen_block(2, 0, 0)
    add_vector_block(2, 2, 2)
    
    add_strassen_block(4, 0, 0)
    add_vector_block(4, 2, 2)
    
    # Last block: p in [6, 7). Size 1.
    # A[:, 0:3], B[0:3, 6:7].
    # Schoolbook 6.
    for i in range(2):
        for j in range(3):
            global_rank = rank_offset
            U[i * 3 + j, global_rank] = 1
            V[j * 7 + 6, global_rank] = 1
            W[6 * 2 + i, global_rank] = 1
            rank_offset += 1
            
    # Check rank_offset
    # 3 * (7 + 4) + 6 = 33 + 6 = 39.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify
    n, m, p = 2, 3, 7
    # Generate random matrices
    np.random.seed(42)
    A = np.random.randint(0, 10, (n, m))
    B = np.random.randint(0, 10, (m, p))
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[k*n+i, :] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])?
    # No, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor is T_{i, j, k} = 1 if we multiply A_{i, j} and B_{j, k}.
    # The decomposition satisfies:
    # sum_r U[i, r] V[j, r] W[k, r] = delta? No.
    # The standard form is:
    # C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) (sum_k V[j, k, r] B[j, k]) ?
    # The interface says:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # This implies:
    # C[i, k] = sum_r W[k*n+i, r] * ( sum_j U[i*m+j, r] * A[i, j] ) * ( sum_l V[j*p+k, r] * B[j, k] )?
    # No, that's not right.
    # Usually, U, V, W are such that:
    # C = sum_r (U_r * A) \circ (V_r * B) ?
    # Let's check the seed program.
    # Seed:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # This corresponds to the term A[i, j] * B[j, k] contributing to C[i, k].
    # So the formula is:
    # C[i, k] = sum_r W[k*n+i, r] * U[i*m+j, r] * A[i, j] * V[j*p+k, r] * B[j, k]
    # Wait, the sum over r is outside.
    # C[i, k] = sum_r W[k*n+i, r] * ( sum_j U[i*m+j, r] A[i, j] ) * ( sum_l V[l*p+k, r] B[l, k] )?
    # No, the indices must match.
    # In the seed, for a fixed r, U is non-zero at (i, j), V is non-zero at (j, k), W is non-zero at (k, i).
    # So for a fixed r, the term is U[i, j, r] V[j, k, r] W[k, i, r] A[i, j] B[j, k].
    # Summing over r, i, j, k?
    # No, C[i, k] is the output.
    # C[i, k] = sum_r W[k, i, r] * ( sum_j U[i, j, r] A[i, j] ) * ( sum_l V[l, k, r] B[l, k] )?
    # This doesn't look like a standard rank decomposition.
    # Standard rank decomposition of the tensor T_{i, j, k}:
    # T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
    # Then C[i, k] = sum_{j, l} T_{i, j, k} A[i, j] B[j, k]? No.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor is T_{i, j, k} = 1 for all i, j, k? No.
    # The multiplication tensor is T_{i, j, k} = 1 if the indices are compatible?
    # Actually, the tensor T is defined such that C[i, k] = sum_{j} T_{i, j, k} A[i, j] B[j, k].
    # Here T_{i, j, k} = 1 for all i, j, k? No, j is summed.
    # The tensor is T_{i, j, k} = 1.
    # Wait, the seed program sets U, V, W to 1 for specific (i, j, k).
    # This implies the decomposition is:
    # C[i, k] = sum_r W[k, i, r] * (sum_j U[i, j, r] A[i, j]) * (sum_l V[l, k, r] B[l, k])?
    # No, the seed sets U[i, j, r]=1, V[j, k, r]=1, W[k, i, r]=1.
    # This implies that for a specific r, we pick a specific (i, j, k) and multiply A[i, j] and B[j, k] and add to C[i, k].
    # So the formula is:
    # C[i, k] = sum_r W[k, i, r] * ( sum_j U[i, j, r] A[i, j] ) * ( sum_l V[l, k, r] B[l, k] ) is WRONG.
    # It should be:
    # C[i, k] = sum_r W[k, i, r] * U[i, j_r, r] * V[j_r, k, r] * A[i, j_r] * B[j_r, k]
    # This requires that for each r, U and V have only one non-zero entry in the j dimension?
    # In the seed, yes.
    # In my Strassen decomposition, U and V have multiple non-zero entries.
    # Does the formula hold?
    # C[i, k] = sum_r W[k, i, r] * ( sum_j U[i, j, r] A[i, j] ) * ( sum_l V[l, k, r] B[l, k] )?
    # If U and V have multiple non-zeros, this computes a bilinear form.
    # Yes, that is the definition of the rank decomposition.
    # So my code should be correct.

    # Compute C_decomp
    C_decomp = np.zeros((n, p))
    for r in range(total_rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_val = W[:, r]
        
        # u_col is length n*m. Reshape to (n, m).
        u_mat = u_col.reshape((n, m))
        # v_col is length m*p. Reshape to (m, p).
        v_mat = v_col.reshape((m, p))
        
        # Term for this r:
        # A_u = A * u_mat (element-wise)
        # B_v = B * v_mat (element-wise)
        # Sum A_u and B_v?
        # No, the formula is:
        # C[i, k] += w_val[k*n+i] * (sum_j u_mat[i, j] A[i, j]) * (sum_l v_mat[l, k] B[l, k])
        
        a_sum = np.sum(u_mat * A, axis=1) # shape (n,)
        b_sum = np.sum(v_mat * B, axis=0) # shape (p,)
        
        # Outer product
        c_term = np.outer(a_sum, b_sum) # shape (n, p)
        
        # w_val is shape (p*n). Reshape to (p, n).
        w_mat = w_val.reshape((p, n)).T # shape (n, p)
        
        C_decomp += c_term * w_mat
        
    if np.allclose(C_expected, C_decomp):
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")
        print(C_expected)
        print(C_decomp)

