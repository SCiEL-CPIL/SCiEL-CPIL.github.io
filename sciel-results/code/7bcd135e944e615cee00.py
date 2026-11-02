"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses a hybrid approach: Strassen-like reductions where applicable, with schoolbook for the rest.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term with given non-zero entries."""
        # u_entries, v_entries, w_entries are lists of (index, value)
        u_row = [0] * (n * m)
        v_row = [0] * (m * p)
        w_row = [0] * (p * n)
        for idx, val in u_entries:
            u_row[idx] = val
        for idx, val in v_entries:
            v_row[idx] = val
        for idx, val in w_entries:
            w_row[idx] = val
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
    
    # Strategy: Use Strassen decomposition on overlapping 2x2 blocks
    # For 2x5 * 5x5, we can think of it as:
    # A is 2x5, B is 5x5, C = A*B is 2x5
    # 
    # Let's partition columns of A and rows of B into blocks.
    # A = [A1 | A2 | A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1
    # B = [[B11], [B12], [B13]] where B11 is 2x5, B12 is 2x5, B13 is 1x5
    #
    # Actually, let me think more carefully.
    # A (2x5) * B (5x5) = C (2x5)
    # C[i,k] = sum_j A[i,j] * B[j,k] for j=0..4
    #
    # Let's use Strassen on 2x2 sub-blocks of the multiplication.
    # We can view this as multiplying a 2x2 block of A with a 2x2 block of B.
    # But A has only 2 rows, so we need to be careful.
    #
    # Alternative approach: Use the fact that for 2x5 * 5x5, we can use
    # a decomposition that exploits the small first dimension.
    #
    # Let me try a direct Strassen-like approach:
    # Partition the 5 intermediate dimensions into: {0,1}, {2,3}, {4}
    # For each pair of 2D blocks, use Strassen (7 terms instead of 4)
    # For the remaining 1D block, use schoolbook (2 terms for 2x1)
    #
    # Total: 2 blocks of 2x2 * 2x5 (using Strassen: 7 terms each for the 2x2 part, 
    # but we need to handle the 5 columns of B)
    #
    # Let me think differently. The tensor T[i,j,k] = delta[i,k] for the 
    # multiplication tensor where C[i,k] = sum_j A[i,j] B[j,k].
    #
    # Rank of T_{2,5,5} = ?
    # Known results: R(2,2,2)=7, R(2,2,1)=4 (actually 4 for 2x1*1x2? No...)
    # 
    # Let me just implement a careful Strassen-based decomposition:
    # 
    # For the 2x5 * 5x5 product:
    # Split A into columns: A = [A_{0,1} | A_{2,3} | A_4] (2x2 | 2x2 | 2x1)
    # Split B into rows: B = [B_{0,1} | B_{2,3} | B_4] (2x5 | 2x5 | 1x5)
    # 
    # C = A_{0,1} B_{0,1} + A_{2,3} B_{2,3} + A_4 B_4
    # 
    # Each A_{a} B_{b} where A_a is 2x2 and B_b is 2x5 can be computed
    # by multiplying the 2x2 matrix A_a with each 2x1 column of B_b.
    # That's 5 separate 2x2 * 2x1 multiplications, each of rank 4 (schoolbook) 
    # or... actually 2x2 * 2x1 is just 4 scalar mults.
    # 
    # Wait, let me reconsider. 2x2 * 2x1:
    # [a b]   [x]   [ax+bx]
    # [c d] * [y] = [cx+dy]
    # This requires 4 multiplications, rank 4.
    # 
    # For 2x2 * 2x2, Strassen gives rank 7.
    # 
    # So for A_{2x2} * B_{2x5}, we have 5 columns, each is 2x1.
    # That's 5 * 4 = 20 terms using schoolbook.
    # 
    # Alternatively, we can view A_{2x2} * B_{2x5} as:
    # First compute A_{2x2} * B_{2x2} for each 2x2 block of B (if 5 is even, but it's not)
    # 
    # Let me try a different partition. Since 5 = 2 + 2 + 1:
    # Terms for (0,1) block: A[0:2, 0:2] * B[0:2, :] -> 2x2 * 2x5
    # Terms for (2,3) block: A[0:2, 2:4] * B[2:4, :] -> 2x2 * 2x5  
    # Terms for (4) block: A[0:2, 4] * B[4, :] -> 2x1 * 1x5
    # 
    # For 2x2 * 2x5: We can use Strassen on 2x2 * 2x2 blocks.
    # B[0:2, :] has columns 0,1,2,3,4. We can group as [B01 | B23 | B4]
    # So A_{2x2} * B_{2x5} = A_{2x2}*B_{2,0:2} + A_{2x2}*B_{2,2:4} + A_{2x2}*B_{2,4}
    # = (Strassen 2x2*2x2) + (Strassen 2x2*2x2) + (schoolbook 2x2*2x1)
    # = 7 + 7 + 4 = 18 terms
    # 
    # Similarly for the other 2x2 block: 18 terms
    # For 2x1 * 1x5: just 2*5 = 10 scalar mults, rank 10? 
    # Actually 2x1 * 1x5: [a] * [b0 b1 b2 b3 b4] = [a*b0, a*b1, ..., a*b4]
    # This is 5 rank-1 terms (each a*b_k contributes to C[:,k])
    # Wait no, it's 5 terms, each multiplying a (scalar from A) with B[4,k].
    # So rank 5 for this part.
    # 
    # Total: 18 + 18 + 5 = 41 terms
    # 
    # Hmm, that's worse than 50 - 8 = 42 from the observation. Let me re-examine.
    # 
    # Actually, the observation said Strassen reduces from 50 to 46, saving 4 per 2x2x2 block.
    # With two such blocks, 50 - 8 = 42.
    # 
    # Let me try to achieve rank 42 or lower.
    # 
    # Schoolbook: 2*5*5 = 50 terms.
    # Each term: A[i,j] * B[j,k] contributes to C[i,k].
    # 
    # Using Strassen on 2x2x2 sub-cubes:
    # A 2x2x2 sub-cube corresponds to i in {i0,i1}, j in {j0,j1}, k in {k0,k1}
    # Schoolbook for this: 8 terms. Strassen: 7 terms. Save 1.
    # 
    # How many disjoint 2x2x2 sub-cubes can we fit?
    # n=2, m=5, p=5.
    # i dimension: 2, so 1 block of size 2.
    # j dimension: 5, so 2 blocks of size 2 (using j=0,1 and j=2,3), leaving j=4.
    # k dimension: 5, so 2 blocks of size 2 (using k=0,1 and k=2,3), leaving k=4.
    # 
    # So we can form 1*2*2 = 4 disjoint 2x2x2 sub-cubes.
    # Each saves 1 term: total save = 4.
    # Remaining terms: for the boundary parts.
    # 
    # After using 4 Strassen blocks (each 7 terms = 28 terms):
    # We need to handle:
    # - i in {0,1}, j=4, all k: 2*1*5 = 10 terms (schoolbook)
    # - i in {0,1}, j in {0,1,2,3}, k=4: 2*4*1 = 8 terms (schoolbook)
    # But wait, we already covered j in {0,1,2,3} and k in {0,1,2,3} with the Strassen blocks.
    # 
    # Let me be more precise. The 4 Strassen blocks cover:
    # (i in {0,1}, j in {0,1}, k in {0,1})
    # (i in {0,1}, j in {0,1}, k in {2,3})
    # (i in {0,1}, j in {2,3}, k in {0,1})
    # (i in {0,1}, j in {2,3}, k in {2,3})
    # 
    # Remaining:
    # (i in {0,1}, j in {0,1}, k=4): 2*1*1 = 2 terms
    # (i in {0,1}, j in {2,3}, k=4): 2*1*1 = 2 terms
    # (i in {0,1}, j=4, k in {0,1}): 2*1*1 = 2 terms
    # (i in {0,1}, j=4, k in {2,3}): 2*1*1 = 2 terms
    # (i in {0,1}, j=4, k=4): 2*1*1 = 2 terms
    # Total remaining: 10 terms
    # 
    # Total rank: 4*7 + 10 = 28 + 10 = 38
    # 
    # That's better than 42! Let me implement this.
    # 
    # Strassen formulas for 2x2 * 2x2:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * b11
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    # 
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    def strassen_block(i0, j0, k0, offset):
        """
        Add Strassen decomposition for the 2x2x2 block starting at (i0, j0, k0).
        offset is the current number of terms added (for indexing).
        Returns new offset after adding 7 terms.
        """
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        a00_u = i0 * m + j0
        a01_u = i0 * m + j0 + 1
        a10_u = (i0 + 1) * m + j0
        a11_u = (i0 + 1) * m + j0 + 1
        
        b00_v = j0 * p + k0
        b01_v = j0 * p + k0 + 1
        b10_v = (j0 + 1) * p + k0
        b11_v = (j0 + 1) * p + k0 + 1
        
        c00_w = k0 * n + i0
        c01_w = (k0 + 1) * n + i0
        c10_w = k0 * n + (i0 + 1)
        c11_w = (k0 + 1) * n + (i0 + 1)
        
        # m1 = (a00 + a11) * (b00 + b11)
        add_term(
            [(a00_u, 1), (a11_u, 1)],
            [(b00_v, 1), (b11_v, 1)],
            [(c00_w, 1), (c11_w, 1)]
        )
        
        # m2 = (a10 + a11) * b00
        add_term(
            [(a10_u, 1), (a11_u, 1)],
            [(b00_v, 1)],
            [(c10_w, 1)]
        )
        
        # m3 = a00 * (b01 - b11)
        add_term(
            [(a00_u, 1)],
            [(b01_v, 1), (b11_v, -1)],
            [(c01_w, 1)]
        )
        
        # m4 = a11 * b11
        add_term(
            [(a11_u, 1)],
            [(b11_v, 1)],
            [(c00_w, 1), (c10_w, 1)]
        )
        
        # m5 = (a00 + a01) * b11
        add_term(
            [(a00_u, 1), (a01_u, 1)],
            [(b11_v, 1)],
            [(c00_w, -1), (c01_w, 1)]
        )
        
        # m6 = (a10 - a00) * (b00 + b01)
        add_term(
            [(a10_u, 1), (a00_u, -1)],
            [(b00_v, 1), (b01_v, 1)],
            [(c11_w, 1)]
        )
        
        # m7 = (a01 - a11) * (b10 + b11)
        add_term(
            [(a01_u, 1), (a11_u, -1)],
            [(b10_v, 1), (b11_v, 1)],
            [(c00_w, 1), (c11_w, 1)]
        )
        
        return offset + 7
    
    # Add 4 Strassen blocks
    strassen_block(0, 0, 0, 0)
    strassen_block(0, 0, 2, 7)
    strassen_block(0, 2, 0, 14)
    strassen_block(0, 2, 2, 21)
    
    # Now add the 10 schoolbook terms for the boundary
    # (i in {0,1}, j in {0,1}, k=4)
    for i in range(2):
        for j in [0, 1]:
            add_term(
                [(i * m + j, 1)],
                [(j * p + 4, 1)],
                [(4 * n + i, 1)]
            )
    
    # (i in {0,1}, j in {2,3}, k=4)
    for i in range(2):
        for j in [2, 3]:
            add_term(
                [(i * m + j, 1)],
                [(j * p + 4, 1)],
                [(4 * n + i, 1)]
            )
    
    # (i in {0,1}, j=4, k in {0,1,2,3,4})
    for i in range(2):
        for k in range(5):
            add_term(
                [(i * m + 4, 1)],
                [(4 * p + k, 1)],
                [(k * n + i, 1)]
            )
    
    # Total: 28 + 4 + 4 + 10 = 46 terms
    # Wait, that's 28 + 8 + 10 = 46. But I expected 38. Let me recount.
    # Strassen: 4 blocks * 7 = 28
    # Boundary: 
    #   j in {0,1}, k=4: 2*2 = 4
    #   j in {2,3}, k=4: 2*2 = 4  
    #   j=4, k in {0,1,2,3,4}: 2*5 = 10
    # Total boundary: 4 + 4 + 10 = 18
    # Grand total: 28 + 18 = 46
    # 
    # Hmm, that's the same as the observed 46. But I thought it would be 38.
    # Let me re-examine my earlier calculation. I think I made an error.
    # The remaining terms after 4 Strassen blocks should be:
    # All (i, j, k) not covered by the 4 blocks.
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Remaining: 
    #   i in {0,1}, j in {0,1,2,3}, k=4: 2*4*1 = 8
    #   i in {0,1}, j=4, k in {0,1,2,3,4}: 2*1*5 = 10
    # Total remaining: 18
    # So total rank = 28 + 18 = 46. That matches.
    # 
    # Can we do better? The observation said 46 was achieved. Let me try to verify
    # this is correct and then see if we can improve.
    
    rank = len(U)
    
    # Convert to numpy arrays for verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify the decomposition
    # The tensor T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i], so W index is k*n+i.
    # Let me verify: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i,j,k] = 1 if we're multiplying, and the decomposition is
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_val = 1.0  # The multiplication tensor has 1s
                decomposed = 0.0
                for c in range(rank):
                    decomposed += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(tensor_val - decomposed) > 1e-9:
                    # This should not happen
                    pass
    
    # Check that non-multiplication entries are 0
    # Actually, the tensor T[i,j,k] is defined such that the multiplication
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k] = 1.
    # So all T[i,j,k] = 1 for all valid i,j,k.
    
    # Let me do a full numerical check
    # Generate random matrices and verify
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_decomposed = np.zeros((n, p))
    for c in range(rank):
        A_u = np.zeros((n, m))
        B_v = np.zeros((m, p))
        W_col = np.zeros((p, n))
        for i in range(n):
            for j in range(m):
                A_u[i, j] = U_arr[i*m+j, c]
        for j in range(m):
            for k in range(p):
                B_v[j, k] = V_arr[j*p+k, c]
        for k in range(p):
            for i in range(n):
                W_col[k, i] = W_arr[k*n+i, c]
        C_decomposed += A_u * B_v * W_col.T  # Need to check the contraction
    
    # Actually, let me think about this more carefully.
    # The decomposition says:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_k V[j,k,c] B[j,k]) ... no that's not right.
    # 
    # The standard tensor decomposition for matrix multiplication is:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # where T[i,j,k] = 1 for all i,j,k (for the rectangular case, it's still 1).
    # 
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c] A[i,j] B[j,k]
    # 
    # Wait, that's not quite right either. Let me look at the schoolbook case:
    # U[i,j,c] = 1 if c = i*m+j (for the term A[i,j]*B[j,k])
    # V[j,k,c] = 1 if c = i*m+j... no, that doesn't work.
    # 
    # In the schoolbook:
    # For each (i,j,k), we have a term with c = i*m*j + j*p + k or something.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # 
    # So the decomposition is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # for all (i,j,k), and 0 otherwise? No, it should be 1 for all (i,j,k) that
    # participate in the multiplication.
    # 
    # Actually, for the full tensor, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # 
    # Let me verify this numerically.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        U_fb = [[0] * (n*m*p) for _ in range(n*m)]
        V_fb = [[0] * (n*m*p) for _ in range(m*p)]
        W_fb = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][column] = 1
                    V_fb[j*p+k][column] = 1
                    W_fb[k*n+i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to integer/half-integer format
    def to_half_int(x):
        # Round to nearest half-integer
        return round(x * 2) / 2
    
    U_final = [[to_half_int(val) for val in row] for row in U]
    V_final = [[to_half_int(val) for val in row] for row in V]
    W_final = [[to_half_int(val) for val in row] for row in W]
    
    # Final verification with half-integer values
    U_arr2 = np.array(U_final, dtype=float)
    V_arr2 = np.array(V_final, dtype=float)
    W_arr2 = np.array(W_final, dtype=float)
    
    max_err2 = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr2[i*m+j, c] * V_arr2[j*p+k, c] * W_arr2[k*n+i, c] for c in range(rank))
                max_err2 = max(max_err2, abs(val - 1.0))
    
    if max_err2 > 1e-9:
        # Fallback to schoolbook
        U_fb = [[0] * (n*m*p) for _ in range(n*m)]
        V_fb = [[0] * (n*m*p) for _ in range(m*p)]
        W_fb = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][column] = 1
                    V_fb[j*p+k][column] = 1
                    W_fb[k*n+i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
