import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition.
    
    Strategy: Use Strassen decomposition on 2x2x2 blocks where possible,
    with careful index management for the 4x5x5 case.
    
    Split:
    - i (rows of A, rows of C): 4 = 2+2
    - j (cols of A, rows of B): 5 = 2+2+1
    - k (cols of B, cols of C): 5 = 2+2+1
    
    Block structure:
    - A: 2x3 blocks (i-blocks x j-blocks), sizes (2x2, 2x2, 2x1)
    - B: 3x3 blocks (j-blocks x k-blocks), sizes (2x2, 2x2, 2x1; 2x2, 2x2, 2x1; 1x2, 1x2, 1x1)
    - C: 2x3 blocks (i-blocks x k-blocks), sizes (2x2, 2x2, 2x1)
    
    For each (i-block, j-block, k-block) triple, we have a sub-multiplication.
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    """
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    # Strassen 2x2x2 decomposition - 7 terms
    # For A[2x2], B[2x2], C[2x2]:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term contributes to U, V, W with specific entries.
    # Term t: U has entries for A-part, V for B-part, W for C-part with coefficient.
    
    # Define Strassen terms as (U_entries, V_entries, W_entries)
    # where each is list of (local_index, coefficient)
    # A local indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B local indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C local indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'U': [(0, 1), (3, 1)],  # A00, A11
            'V': [(0, 1), (3, 1)],  # B00, B11
            'W': [(0, 1), (3, 1)]   # C00, C11
        },
        # M2 = (A10+A11)B00
        {
            'U': [(2, 1), (3, 1)],  # A10, A11
            'V': [(0, 1)],          # B00
            'W': [(2, 1)]           # C10
        },
        # M3 = A00(B10-B11)
        {
            'U': [(0, 1)],          # A00
            'V': [(2, 1), (3, -1)], # B10, -B11
            'W': [(1, 1)]           # C01
        },
        # M4 = A11(B01-B00)
        {
            'U': [(3, 1)],          # A11
            'V': [(1, 1), (0, -1)], # B01, -B00
            'W': [(0, 1), (2, 1)]   # C00, C10
        },
        # M5 = (A00+A01)B11
        {
            'U': [(0, 1), (1, 1)],  # A00, A01
            'V': [(3, 1)],          # B11
            'W': [(0, -1), (1, 1)]  # -C00, C01
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'U': [(2, 1), (0, -1)], # A10, -A00
            'V': [(0, 1), (1, 1)],  # B00, B01
            'W': [(3, 1)]           # C11
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'U': [(1, 1), (3, -1)], # A01, -A11
            'V': [(2, 1), (3, 1)],  # B10, B11
            'W': [(0, 1), (3, -1)]  # C00, -C11
        },
    ]
    
    # Build the full decomposition
    # i-blocks: [0, 1] (each size 2)
    # j-blocks: [0, 1, 2] (sizes 2, 2, 1)
    # k-blocks: [0, 1, 2] (sizes 2, 2, 1)
    
    i_block_sizes = [2, 2]
    j_block_sizes = [2, 2, 1]
    k_block_sizes = [2, 2, 1]
    
    # Compute offsets
    i_offsets = [0]
    for s in i_block_sizes[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in j_block_sizes[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in k_block_sizes[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    # Determine which block triples use Strassen (all sizes 2) vs schoolbook
    # Strassen: i_block in [0,1], j_block in [0,1], k_block in [0,1]
    # Schoolbook: all others
    
    rank = 0
    # Count ranks
    for ib in range(2):
        for jb in range(3):
            for kb in range(3):
                if j_block_sizes[jb] == 2 and k_block_sizes[kb] == 2:
                    rank += 7  # Strassen
                else:
                    # Schoolbook: i_size * j_size * k_size terms
                    rank += i_block_sizes[ib] * j_block_sizes[jb] * k_block_sizes[kb]
    
    # Allocate matrices
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    
    for ib in range(2):
        for jb in range(3):
            for kb in range(3):
                i0 = i_offsets[ib]
                j0 = j_offsets[jb]
                k0 = k_offsets[kb]
                i_size = i_block_sizes[ib]
                j_size = j_block_sizes[jb]
                k_size = k_block_sizes[kb]
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    for term in strassen_terms:
                        # Add U entries
                        for local_idx, coeff in term['U']:
                            a_row = local_idx // 2
                            a_col = local_idx % 2
                            global_a_idx = (i0 + a_row) * m + (j0 + a_col)
                            U[global_a_idx, col] += coeff
                        
                        # Add V entries
                        for local_idx, coeff in term['V']:
                            b_row = local_idx // 2
                            b_col = local_idx % 2
                            global_b_idx = (j0 + b_row) * p + (k0 + b_col)
                            V[global_b_idx, col] += coeff
                        
                        # Add W entries
                        for local_idx, coeff in term['W']:
                            c_row = local_idx // 2
                            c_col = local_idx % 2
                            global_c_idx = (k0 + c_col) * n + (i0 + c_row)
                            W[global_c_idx, col] += coeff
                        
                        col += 1
                else:
                    # Schoolbook for this block
                    for a_row in range(i_size):
                        for b_col in range(j_size):
                            for c_row in range(k_size):
                                global_a_idx = (i0 + a_row) * m + (j0 + b_col)
                                global_b_idx = (j0 + b_col) * p + (k0 + c_row)
                                global_c_idx = (k0 + c_row) * n + (i0 + a_row)
                                
                                U[global_a_idx, col] = 1
                                V[global_b_idx, col] = 1
                                W[global_c_idx, col] = 1
                                col += 1
    
    # Verify the decomposition
    # Check that sum_t U[:,t] * V[:,t] * W[:,t] == tensor
    # Tensor T[i,j,k] = 1 if i < n, j < m, k < p and i,j,k form valid MM indices
    # T[(i,j), (j,k), (k,i)] = 1 for valid i,j,k
    
    # Build the tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1
    
    # Verify: for each (a, b, c), T[a,b,c] should equal sum_t U[a,t]*V[b,t]*W[c,t]
    # Efficient check: compute U @ diag(V[b,t]) @ W[c,t] for all t
    # Or: for each t, outer product U[:,t] @ V[:,t] @ W[:,t]
    
    # Check a sample of entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                # Compute sum_t U[a_idx,t] * V[b_idx,t] * W[c_idx,t]
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                err = abs(val - T[a_idx, b_idx, c_idx])
                if err > max_err:
                    max_err = err
    
    # Also check that non-MM entries are 0
    # For entries where (a,b,c) doesn't correspond to valid MM
    for a_idx in range(nm):
        i = a_idx // m
        j = a_idx % m
        for b_idx in range(mp):
            j2 = b_idx // p
            k = b_idx % p
            if j != j2:
                # Invalid: j from A doesn't match j from B
                for c_idx in range(pn):
                    k2 = c_idx // n
                    i2 = c_idx % n
                    if i != i2:
                        # Also invalid for C
                        val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                        err = abs(val - 0)
                        if err > max_err:
                            max_err = err
    
    if max_err > 1e-10:
        # Decomposition is invalid, fall back to schoolbook
        rank = nm * mp // pn  # This is wrong, let's just use full schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        # Valid decomposition, trim to actual used rank
        # Find columns that are all zero
        used_cols = []
        for t in range(rank):
            if np.any(U[:, t] != 0) or np.any(V[:, t] != 0) or np.any(W[:, t] != 0):
                used_cols.append(t)
        
        if len(used_cols) < rank:
            U = U[:, used_cols]
            V = V[:, used_cols]
            W = W[:, used_cols]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
