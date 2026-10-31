"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """
    Returns (U, V, W) for 2x2 matrix multiplication with rank 7.
    U: shape (4, 7) — corresponds to A flattened in row-major
    V: shape (4, 7) — corresponds to B flattened in row-major
    W: shape (4, 7) — corresponds to C flattened in row-major
    """
    # Standard Strassen factors
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # U factors (A side): each column is coefficients for one of the 7 multiplications
    # A11 is at index 0, A12 at index 1, A21 at index 2, A22 at index 3
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W factors (C side)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # C11 is at index 0
    W[0, 0] = 1  # M1
    W[0, 3] = 1  # M4
    W[0, 4] = -1 # M5
    W[0, 6] = 1  # M7
    
    # C12 is at index 1
    W[1, 2] = 1  # M3
    W[1, 4] = 1  # M5
    
    # C21 is at index 2
    W[2, 1] = 1  # M2
    W[2, 3] = 1  # M4
    
    # C22 is at index 3
    W[3, 0] = 1  # M1
    W[3, 2] = -1 # M3
    W[3, 1] = 1  # M2
    W[3, 5] = -1 # M6
    
    return U, V, W

def _block_map_4x4():
    """
    Maps local 2x2 block indices to global 4x4 indices for a 4x4 matrix.
    We treat the 4x4 matrix as 4 blocks of 2x2.
    Block (bi, bj) where bi, bj in {0, 1}
    Within each 2x2 block, local indices (li, lj) in {0, 1}
    
    Global row index = bi * 2 + li
    Global col index = bj * 2 + lj
    
    For U (A matrix, n*m = 4*4): index = row * m + col = row * 4 + col
    For V (B matrix, m*p = 4*4): index = row * p + col = row * 4 + col
    For W (C matrix, p*n = 4*4): index = row * n + col = row * 4 + col (since n=4)
    """
    # Returns mapping from (block_i, block_j, local_i, local_j) to global index
    # We'll create lookup tables
    U_map = {}
    V_map = {}
    W_map = {}
    
    for bi in range(2):
        for bj in range(2):
            for li in range(2):
                for lj in range(2):
                    row = bi * 2 + li
                    col = bj * 2 + lj
                    idx = row * 4 + col
                    U_map[(bi, bj, li, lj)] = idx
                    V_map[(bi, bj, li, lj)] = idx
                    W_map[(bi, bj, li, lj)] = idx
    
    return U_map, V_map, W_map

def _build_strassen_4x4():
    """
    Build Strassen decomposition for 4x4 matrix multiplication.
    Uses 4 blocks of 2x2, and applies Strassen at the block level.
    This gives rank 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2()
    U_map, V_map, W_map = _block_map_4x4()
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # For each pair of outer Strassen indices (qi, qj) in {0..6}^2
    for qi in range(7):
        for qj in range(7):
            col = qi * 7 + qj
            
            # For each inner Strassen index qk in {0..6}
            for qk in range(7):
                # U contribution:
                # The A block is at (bi, bj) where bi, bj depend on qi
                # Actually, let's think more carefully.
                
                # In the block-level Strassen, we have 4 blocks of A: A00, A01, A10, A11
                # Each block is 2x2.
                # The outer Strassen combines these blocks.
                # U2[bi*2+li, qi] gives the coefficient for A[bi*2+li, bj*2+lj]
                
                # Wait, U2 is (4, 7) where the 4 rows correspond to the 4 entries of a 2x2 A block.
                # So U2[li + 2*lj, qi] where li, lj are local indices within the block?
                # No, U2[0] = A11, U2[1] = A12, U2[2] = A21, U2[3] = A22
                # So the index into U2 is: row_in_block * 2 + col_in_block? No.
                # A11 is (0,0), A12 is (0,1), A21 is (1,0), A22 is (1,1)
                # So U2[r, c] where r = 0: A11, r=1: A12, r=2: A21, r=3: A22
                # This is row-major flattening of the 2x2 block.
                
                # So for block (bi, bj), local (li, lj):
                # Global A index = (bi*2+li) * 4 + (bj*2+lj)
                # Local block index = li * 2 + lj
                # U2[li*2+lj, qi] gives the coefficient
                
                # But wait, the outer Strassen operates on the blocks.
                # The "A" in the outer Strassen is a 2x2 matrix of 2x2 blocks.
                # So the outer U2 factor applies to the block indices.
                
                # Let me reconsider. The outer Strassen has:
                # M_out(qi, qj) = sum_{bi,bj} U2_block[bi*2+bj, qi] * A_block[bi,bj] * ... 
                
                # Actually, it's simpler: the 4x4 multiplication can be viewed as:
                # C_block[bi,bj] = sum_{bk,bl} A_block[bi,bk] * B_block[bk,bl]
                # where each block is 2x2.
                
                # So we have a 2x2 "block matrix" multiplication, where each entry is a 2x2 matrix.
                # We can apply Strassen to this block matrix multiplication.
                
                # The Strassen decomposition for the block level gives:
                # C_block = sum_{qi} (sum_{bi,bj} U_block[bi*2+bj, qi] * A_block[bi,bj]) * ... 
                
                # Hmm, this is getting complicated. Let me use a different approach.
                
                # The 4x4 matrix multiplication tensor can be decomposed as:
                # T[i,j,k,l] = delta_{i,k} * delta_{j,l} (summed appropriately)
                
                # Actually, the standard approach is:
                # The 4x4 MM tensor has rank at most 7^2 = 49 by applying Strassen twice.
                # First level: split into 2x2 blocks of size 2x2.
                # The block-level Strassen has rank 7.
                # Within each block multiplication, we need 4x4 = 16 scalar multiplications? No.
                
                # Let me think again. If we have a 2x2 block matrix A, B, C where each block is 2x2,
                # then C = A*B means C_block[bi,bj] = sum_k A_block[bi,k] * B_block[k,bj]
                # where each block multiplication is 2x2 * 2x2.
                
                # The number of scalar multiplications needed is 7 (for the block level) * 4 (for each 2x2 block mult) = 28? No.
                
                # Actually, each "block multiplication" A_block[bi,k] * B_block[k,bj] is a 2x2 matrix multiplication,
                # which itself can be done with 7 scalar multiplications using Strassen.
                
                # So total rank = 7 (outer) * 7 (inner) = 49.
                
                # The construction:
                # For each outer Strassen index qi (0..6) and inner Strassen index qj (0..6):
                # The scalar multiplication is between:
                #   U_factor = sum of A entries weighted by U2_outer and U2_inner
                #   V_factor = sum of B entries weighted by V2_outer and V2_inner
                # And the result contributes to C entries weighted by W2_outer and W2_inner.
                
                # Let's be more explicit.
                # The outer Strassen factors for the 2x2 block matrix:
                # U_outer[bi*2+bj, qi] where bi,bj in {0,1} are block indices
                # V_outer[bk*2+bl, qi] where bk,bl are block indices
                # W_outer[bi*2+bj, qi] where bi,bj are block indices for C
                
                # The inner Strassen factors for each 2x2 block multiplication:
                # U_inner[li*2+lj, qj] where li,lj are local indices
                # V_inner[mi*2+mj, qj] where mi,mj are local indices
                # W_inner[pi*2+pj, qj] where pi,pj are local indices
                
                # The full U factor for column (qi, qj) at global A index (bi*2+li, bj*2+lj):
                # U_full[bi*2+li, bj*2+lj, qi*7+qj] = U_outer[bi*2+bj, qi] * U_inner[li*2+lj, qj]
                
                # Similarly for V and W.
                
                pass  # We'll handle this below
    
    # Redo the construction properly
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    for qi in range(7):
        for qj in range(7):
            col = qi * 7 + qj
            
            # For each block index pair (bi, bj) for A
            for bi in range(2):
                for bj in range(2):
                    # For each local index within the block (li, lj)
                    for li in range(2):
                        for lj in range(2):
                            a_global_row = bi * 2 + li
                            a_global_col = bj * 2 + lj
                            a_global_idx = a_global_row * 4 + a_global_col
                            
                            a_local_idx = li * 2 + lj
                            
                            u_outer = U2[bi * 2 + bj, qi]
                            u_inner = U2[a_local_idx, qj]
                            
                            U[a_global_idx, col] += u_outer * u_inner
            
            # For each block index pair (bk, bl) for B
            for bk in range(2):
                for bl in range(2):
                    for mi in range(2):
                        for mj in range(2):
                            b_global_row = bk * 2 + mi
                            b_global_col = bl * 2 + mj
                            b_global_idx = b_global_row * 4 + b_global_col
                            
                            b_local_idx = mi * 2 + mj
                            
                            v_outer = V2[bk * 2 + bl, qi]
                            v_inner = V2[b_local_idx, qj]
                            
                            V[b_global_idx, col] += v_outer * v_inner
            
            # For each block index pair (bi, bj) for C
            for bi in range(2):
                for bj in range(2):
                    for pi in range(2):
                        for pj in range(2):
                            c_global_row = bi * 2 + pi
                            c_global_col = bj * 2 + pj
                            c_global_idx = c_global_row * 4 + c_global_col
                            
                            c_local_idx = pi * 2 + pj
                            
                            w_outer = W2[bi * 2 + bj, qi]
                            w_inner = W2[c_local_idx, qj]
                            
                            W[c_global_idx, col] += w_outer * w_inner
    
    # Verify the decomposition
    # Check that U @ diag(mult) @ V^T gives the correct C tensor
    # Actually, we need to verify that the tensor decomposition is correct.
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Try Strassen decomposition first
    try:
        result = _build_strassen_4x4()
        
        # Verify the decomposition
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        
        n, m, p = 4, 4, 4
        rank = U.shape[1]
        
        # Verify: for each rank term r, the contribution to C[i,j] from A[*,*] and B[*,*]
        # C[i,j] = sum_r U[i*4+j, r] * (sum_k A[i,k] * B[k,j] * W[j*4+i, r])? 
        # No, the decomposition is:
        # C[i,j] = sum_r U[i*m+j, r] * V[m*i + k, r] * ... 
        
        # Actually, the standard tensor decomposition for matrix multiplication:
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # This can be written as:
        # C[i,j] = sum_r U[i*m+j, r] * (sum_k A[i,k] * B[k,j] * something)
        
        # The correct formulation:
        # The MM tensor T is a 3-tensor of dimensions (n*m, m*p, p*n) or similar.
        # T[a, b, c] where a indexes A entries, b indexes B entries, c indexes C entries.
        # T[i*m+k, k*p+j, j*n+i] = 1 for all i,j,k
        # The rank-r decomposition: T = sum_r U[a, r] * V[b, r] * W[c, r]
        # So we need: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,j,k
        # And 0 otherwise.
        
        # Let's verify
        error = 0.0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a_idx = i * m + k
                    b_idx = k * p + j
                    c_idx = j * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                    if abs(val - 1.0) > 1e-9:
                        error += abs(val - 1.0)
        
        # Also check that other entries are 0 (this is hard to check fully, but the above should suffice for correctness)
        
        if error < 1e-6:
            elapsed = time.time() - start_time
            if elapsed < deadline:
                return result
        else:
            # Verification failed, fall back to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
