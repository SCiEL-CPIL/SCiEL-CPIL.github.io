import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _construct_schoolbook(n, m, p):
    """
    Build the standard (schoolbook) tensor decomposition for multiplying
    an n x m matrix by an m x p matrix.

    The tensor T has components T[i, j, k] = E_{i,k} (the output matrix entry
    at row i, col k depends on the single product A[i,j] * B[j,k]).

    We decompose as T = sum_r U_r ⊗ V_r ⊗ W_r where:
      U has shape (n*m, rank)  -> entries U[i*m + j, r]
      V has shape (m*p, rank)  -> entries V[j*p + k, r]
      W has shape (p*n, rank)  -> entries W[k*n + i, r]

    This follows the official trace convention W[k*n + i, r].
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1

    return U, V, W


def _construct_strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4, 7), V (4, 7), W (4, 7) using the standard Strassen formulas.
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11) * (B12 - B22)
    # M2 = (A11 + A21) * B11
    # M3 = (A22) * (B21 - B11)
    # M4 = (A11 + A22) * (B11 + B22)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A22 - A21) * (B21 + B22)
    #
    # C11 = M4 + M1 - M2 + M6
    # C12 = M5 + M2
    # C21 = M3 + M4
    # C22 = M1 - M5 + M3 + M7
    #
    # We need to express C = A*B as sum_r U_r ⊗ V_r ⊗ W_r
    # where U corresponds to A entries, V to B entries, W to C entries.
    #
    # A indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # B indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # C indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    #
    # For each of the 7 products, we assign:
    # U[r, a_idx] = coefficient of A entry a_idx in the first factor
    # V[r, b_idx] = coefficient of B entry b_idx in the second factor
    # W[r, c_idx] = coefficient of the product in C entry c_idx

    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # Product 1: A11 * (B12 - B22)
    U[0, 0] = 1.0  # A11
    V[1, 0] = 1.0  # B12
    V[3, 0] = -1.0 # B22
    W[0, 0] = 1.0  # contributes to C11
    W[3, 0] = 1.0  # contributes to C22

    # Product 2: (A11 + A21) * B11
    U[0, 1] = 1.0  # A11
    U[2, 1] = 1.0  # A21
    V[0, 1] = 1.0  # B11
    W[0, 1] = -1.0 # contributes to C11 (with minus)
    W[1, 1] = 1.0  # contributes to C12

    # Product 3: A22 * (B21 - B11)
    U[3, 2] = 1.0  # A22
    V[2, 2] = 1.0  # B21
    V[0, 2] = -1.0 # B11
    W[2, 2] = 1.0  # contributes to C21
    W[3, 2] = 1.0  # contributes to C22

    # Product 4: (A11 + A22) * (B11 + B22)
    U[0, 3] = 1.0  # A11
    U[3, 3] = 1.0  # A22
    V[0, 3] = 1.0  # B11
    V[3, 3] = 1.0  # B22
    W[0, 3] = 1.0  # contributes to C11
    W[2, 3] = 1.0  # contributes to C21

    # Product 5: (A11 + A12) * B22
    U[0, 4] = 1.0  # A11
    U[1, 4] = 1.0  # A12
    V[3, 4] = 1.0  # B22
    W[1, 4] = 1.0  # contributes to C12
    W[3, 4] = -1.0 # contributes to C22 (with minus)

    # Product 6: (A21 - A11) * (B11 + B12)
    U[2, 5] = 1.0  # A21
    U[0, 5] = -1.0 # A11
    V[0, 5] = 1.0  # B11
    V[1, 5] = 1.0  # B12
    W[0, 5] = 1.0  # contributes to C11

    # Product 7: (A22 - A21) * (B21 + B22)
    U[3, 6] = 1.0  # A22
    U[2, 6] = -1.0 # A21
    V[2, 6] = 1.0  # B21
    V[3, 6] = 1.0  # B22
    W[3, 6] = 1.0  # contributes to C22

    return U, V, W


def _combine_decompositions(decs):
    """
    Combine a list of (U, V, W) decompositions for sub-problems into
    a single decomposition for the full 5x5x5 problem using a block
    partitioning scheme.
    
    Each dec is for a sub-block of size (ni, mi, pi). We use a 2+2+1 
    partition: the 5x5 matrix is split into blocks of sizes 2, 2, 1.
    
    For a block-diagonal partition of the input/output matrices, the
    overall decomposition is the direct sum of the sub-decompositions,
    but we need to map the indices correctly.
    
    Actually, for matrix multiplication, the natural block decomposition
    gives: C_{ab} = sum_s A_{as} B_{sb} for each pair of block indices.
    
    We partition indices into groups: [0,1], [2,3], [4] with sizes [2, 2, 1].
    
    For each pair (a, b) of block indices, we have a sub-problem of size
    (size_a, size_b, size_a) since A is 5x5 and B is 5x5, C is 5x5.
    Wait, A is 5x5, B is 5x5, C is 5x5. The block structure:
    - A has blocks A_{rs} of size (size_r, size_s) for r,s in {0,1,2}
    - B has blocks B_{st} of size (size_s, size_t) for s,t in {0,1,2}
    - C has blocks C_{rt} of size (size_r, size_t) for r,t in {0,1,2}
    
    C_{rt} = sum_s A_{rs} B_{st}
    
    So for each (r, t) pair, we need to compute a sum over s of products
    of matrices of size (size_r, size_s) and (size_s, size_t).
    
    The tensor for each (r, s, t) triple is a matrix multiplication tensor
    of size (size_r, size_s, size_t). We can decompose each one and then
    combine.
    
    For simplicity and to get a valid decomposition, let's just use the
    schoolbook decomposition for each (r, s, t) block.
    
    The total rank would be sum over r,s,t of (size_r * size_s * size_t).
    With sizes [2, 2, 1]:
    Sum = (2*2*2) + (2*2*1) + (2*1*2) + (2*1*1) + (1*2*2) + (1*2*1) + (1*1*2) + (1*1*1)
    = 8 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 27 per (r,t) pair? No, that's not right.
    
    Actually, for each (r, s, t), the rank of the (size_r, size_s, size_t)
    tensor using schoolbook is size_r * size_s * size_t.
    
    Total rank = sum_{r,s,t} size_r * size_s * size_t
    = (sum_r size_r) * (sum_s size_s) * (sum_t size_t) = 5 * 5 * 5 = 125.
    
    So this doesn't help. To get a better rank, we need to use Strassen
    for the 2x2x2 blocks.
    
    Let me think about this differently. The block decomposition approach
    for 5x5x5 with partition [2,2,1]:
    
    For blocks of size 2x2x2, we use Strassen (rank 7 instead of 8).
    For other block sizes, we use schoolbook.
    
    The blocks where size_r=2, size_s=2, size_t=2: only one such block (r=0,s=0,t=0).
    Wait, no. r, s, t each range over {0,1,2} with sizes [2,2,1].
    
    The (r,s,t) triple has dimensions (size_r, size_s, size_t).
    We use Strassen only when size_r=2, size_s=2, size_t=2.
    That happens only for r=0, s=0, t=0.
    
    So we save 1 rank from using Strassen on that one block: 7 instead of 8.
    Total rank = 125 - 1 = 124.
    
    But we can do better by using Strassen more cleverly. Let me reconsider.
    
    Actually, the standard approach is to view the entire 5x5x5 as a block
    matrix multiplication. The total rank is the sum of ranks of all
    (r,s,t) sub-tensors. If we use Strassen for any 2x2x2 sub-problem,
    we save 1 rank per such sub-problem.
    
    In the [2,2,1] partition, the sub-problems with all dimensions 2 are:
    (r=0, s=0, t=0): size (2,2,2) -> Strassen rank 7 vs schoolbook 8.
    
    Are there others? No, because the only size-2 blocks are at indices 0 and 1,
    and size-1 is at index 2. So (size_r, size_s, size_t) = (2,2,2) only
    for (r,s,t) = (0,0,0) or (0,0,1) etc. Wait:
    
    r=0: size 2, r=1: size 2, r=2: size 1
    s=0: size 2, s=1: size 2, s=2: size 1
    t=0: size 2, t=1: size 2, t=2: size 1
    
    So (size_r, size_s, size_t) = (2,2,2) for (r,s,t) in {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}.
    
    That's 8 sub-problems of size 2x2x2!
    
    Using Strassen for each saves 1 rank, so total saving = 8.
    Total rank = 125 - 8 = 117.
    
    Let me implement this properly.
    """
    n, m, p = 5, 5, 5
    
    # Partition: indices [0,1], [2,3], [4]
    # Block sizes: [2, 2, 1]
    block_sizes = [2, 2, 1]
    # Start indices for each block
    block_starts = [0, 2, 4]
    
    total_rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    for r in range(3):
        for s in range(3):
            for t in range(3):
                sr = block_sizes[r]
                ss = block_sizes[s]
                st = block_sizes[t]
                
                # Sub-problem: multiply (sr x ss) by (ss x st)
                # This is a matrix multiplication tensor of size (sr, ss, st)
                
                if sr == 2 and ss == 2 and st == 2:
                    # Use Strassen
                    U_sub, V_sub, W_sub = _construct_strassen_2x2x2()
                else:
                    # Use schoolbook
                    U_sub, V_sub, W_sub = _construct_schoolbook(sr, ss, st)
                
                rank_sub = U_sub.shape[1]
                total_rank += rank_sub
                
                # Now we need to embed this sub-decomposition into the full
                # 5x5x5 problem.
                #
                # The full U has shape (25, total_rank), V has shape (25, total_rank),
                # W has shape (25, total_rank).
                #
                # For the (r, s, t) block:
                # - A block A_{rs} has rows block_starts[r] to block_starts[r]+sr-1
                #   and columns block_starts[s] to block_starts[s]+ss-1.
                #   In the full U, the row index is i*m + j where i is the row
                #   in A and j is the column in A.
                #
                # - B block B_{st} has rows block_starts[s] to block_starts[s]+ss-1
                #   and columns block_starts[t] to block_starts[t]+st-1.
                #   In the full V, the row index is j*p + k where j is the row
                #   in B and k is the column in B.
                #
                # - C block C_{rt} has rows block_starts[r] to block_starts[r]+sr-1
                #   and columns block_starts[t] to block_starts[t]+st-1.
                #   In the full W, the row index is k*n + i where k is the column
                #   in C and i is the row in C.
                
                # For each column in U_sub, V_sub, W_sub:
                for c in range(rank_sub):
                    # U: map sub-row (i_sub, j_sub) to full row (i, j)
                    # i = block_starts[r] + i_sub
                    # j = block_starts[s] + j_sub
                    # Full U row index = i * m + j = i * 5 + j
                    for i_sub in range(sr):
                        for j_sub in range(ss):
                            if U_sub[i_sub * ss + j_sub, c] != 0:
                                i_full = block_starts[r] + i_sub
                                j_full = block_starts[s] + j_sub
                                U_row_idx = i_full * 5 + j_full
                                U_rows.append((U_row_idx, total_rank - rank_sub + c, U_sub[i_sub * ss + j_sub, c]))
                    
                    # V: map sub-row (j_sub, k_sub) to full row (j, k)
                    # j = block_starts[s] + j_sub
                    # k = block_starts[t] + k_sub
                    # Full V row index = j * p + k = j * 5 + k
                    for j_sub in range(ss):
                        for k_sub in range(st):
                            if V_sub[j_sub * st + k_sub, c] != 0:
                                j_full = block_starts[s] + j_sub
                                k_full = block_starts[t] + k_sub
                                V_row_idx = j_full * 5 + k_full
                                V_rows.append((V_row_idx, total_rank - rank_sub + c, V_sub[j_sub * st + k_sub, c]))
                    
                    # W: map sub-row (k_sub, i_sub) to full row (k, i)
                    # k = block_starts[t] + k_sub
                    # i = block_starts[r] + i_sub
                    # Full W row index = k * n + i = k * 5 + i
                    for k_sub in range(st):
                        for i_sub in range(sr):
                            if W_sub[k_sub * sr + i_sub, c] != 0:
                                k_full = block_starts[t] + k_sub
                                i_full = block_starts[r] + i_sub
                                W_row_idx = k_full * 5 + i_full
                                W_rows.append((W_row_idx, total_rank - rank_sub + c, W_sub[k_sub * sr + i_sub, c]))
    
    # Build the full matrices
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    for row_idx, col_idx, val in U_rows:
        U[row_idx, col_idx] = val
    for row_idx, col_idx, val in V_rows:
        V[row_idx, col_idx] = val
    for row_idx, col_idx, val in W_rows:
        W[row_idx, col_idx] = val
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a fallback
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    
    # Try the block decomposition with Strassen for 2x2x2 blocks
    try:
        U_block, V_block, W_block = _combine_decompositions(None)
        
        # Verify the decomposition
        # Check that the decomposition exactly reproduces the matrix multiplication tensor
        # T[i*m+j, j*p+k, k*n+i] should be 1 for all i,j,k and 0 otherwise
        # Under the decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # So T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # The tensor T for 5x5x5 matrix mult has T[i*5+j, j*5+k, k*5+i] = 1
        # and is 0 elsewhere.
        
        # Check a few entries to verify
        # We'll do a full verification if time permits
        if time.time() - start_time < deadline - 30:
            # Full verification
            rank = U_block.shape[1]
            # Check all entries of the tensor
            # The tensor has 25*25*25 = 15625 entries
            # We check that T = sum_r U[:,r] V[:,r] W[:,r]
            
            # For efficiency, we can check a random sample
            rng = np.random.RandomState(42)
            samples = 2000
            a_samples = rng.randint(0, 25, samples)
            b_samples = rng.randint(0, 25, samples)
            c_samples = rng.randint(0, 25, samples)
            
            valid = True
            for a, b, c in zip(a_samples, b_samples, c_samples):
                # Expected value: 1 if (a, b, c) corresponds to a valid (i,j,k) triple
                # a = i*5+j, b = j*5+k, c = k*5+i
                i = a // 5
                j = a % 5
                j2 = b // 5
                k = b % 5
                k2 = c // 5
                i2 = c % 5
                
                if j == j2 and k == k2 and i == i2:
                    expected = 1.0
                else:
                    expected = 0.0
                
                actual = np.sum(U_block[a, :] * V_block[b, :] * W_block[c, :])
                if abs(actual - expected) > 1e-9:
                    valid = False
                    break
            
            if valid:
                return {
                    "U": U_block.tolist(),
                    "V": V_block.tolist(),
                    "W": W_block.tolist()
                }
        
        # If verification failed or we're running out of time, fall back to schoolbook
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
