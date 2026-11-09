import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against identity: T[i,j,k] should be 1 if i,j,k are consistent with matrix mult
    # Actually, the tensor for matrix mult is: T[i,j,k] = delta_{i,j,k} in the sense that
    # C[i,k] = sum_j A[i,j] * B[j,k], so the tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, no. The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This corresponds to a 3-way tensor T where T[i,j,k] = 1 (it's just the identity structure)
    # The decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since every triple contributes to the product)
    # Actually no - the tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in the valid range.
    
    # Let's verify by checking that for random A and B, the reconstructed C matches A@B
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct C from the decomposition
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        u = U[:, r]  # shape (n*m,)
        v = V[:, r]  # shape (m*p,)
        w = W[:, r]  # shape (p*n,)
        
        # u[i*m + j] = U[i,j,r]
        # v[j*p + k] = V[j,k,r]  
        # w[k*n + i] = W[k,i,r]
        
        for i in range(n):
            for j in range(m):
                uij = u[i * m + j]
                if abs(uij) < 1e-15:
                    continue
                for k in range(p):
                    vjk = v[j * p + k]
                    if abs(vjk) < 1e-15:
                        continue
                    wki = w[k * n + i]
                    C_reconstructed[i, k] += uij * vjk * wki
    
    if not np.allclose(C_reconstructed, C_expected, atol=tol):
        return False
    return True

def _to_rational_array(arr, max_denom=1000):
    """Convert float array to rational (as fractions or keep as float if close to integer/half-integer)."""
    # For this problem, we expect integer or half-integer entries
    # Round to nearest 0.5
    rounded = np.round(arr * 2) / 2.0
    if np.allclose(arr, rounded, atol=1e-10):
        return rounded
    # Try integers
    rounded_int = np.round(arr)
    if np.allclose(arr, rounded_int, atol=1e-10):
        return rounded_int
    return arr

def _try_block_decomposition(n, m, p, time_limit=100):
    """
    Try to find a lower-rank decomposition using block structure.
    
    Strategy: Decompose into 2x2 blocks of A and 2x3 blocks of B.
    For each 2x2 x 2x3 subproblem, use rank-6 schoolbook.
    But share computations across the three column groups.
    
    For fixed row-block i (i=0,1), we have:
    - A_i0 (2x2) and A_i1 (2x2)
    - Need to compute A_i0 * B_j0 + A_i1 * B_j1 for j=0,1,2
    
    For each j, this is a 2x2 x 2x3 product.
    
    Key insight: For fixed i, we can compute:
    - M_j = A_i0 * B_j0 + A_i1 * B_j1 for j=0,1,2
    
    This can be seen as: for each j, we do a 2x2 x 2x3 multiplication.
    Total for fixed i: 3 * 6 = 18 multiplications (schoolbook).
    For both i: 36 multiplications.
    
    But we can do better! Notice that B_j0 and B_j1 are different for each j,
    but A_i0 and A_i1 are shared across j.
    
    Alternative: Compute A_i0 * [B_00, B_01, B_02] and A_i1 * [B_10, B_11, B_12]
    where [B_00, B_01, B_02] is a 2x9 matrix (row block 0 of B).
    
    A_i0 * B_row0 is a 2x2 x 2x9 product. Rank at most 18.
    A_i1 * B_row1 is a 2x2 x 2x9 product. Rank at most 18.
    Total: 36.
    
    Hmm, same as before. Let me think differently.
    
    Actually, the standard schoolbook for 4x4 x 4x9 is 144.
    Using 2x2 blocks: we have 4 blocks of A, 6 blocks of B.
    C has 4 blocks (2x2 grid of 2x3 blocks).
    Each C_block = sum of 2 products of (2x2)*(2x3).
    Each (2x2)*(2x3) has rank 6.
    Total: 4 blocks * 2 terms * 6 = 48.
    
    Wait, that's much better than 144! Let me verify this logic.
    
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x3
    
    C = A * B = [[C00, C01, C02], [C10, C11, C12]]
    where Cij is 2x3.
    
    C00 = A00*B00 + A01*B10  (each term is 2x2 * 2x3 = 2x3)
    C01 = A00*B01 + A01*B11
    C02 = A00*B02 + A01*B12
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    C12 = A10*B02 + A11*B12
    
    Each (2x2)*(2x3) product requires 6 scalar multiplications (schoolbook).
    There are 12 such products total (6 C blocks * 2 terms each).
    Total: 12 * 6 = 72 scalar multiplications.
    
    So rank 72 should work! That's a big improvement over 144.
    
    Let me construct this decomposition.
    """
    start_time = time.time()
    
    # We'll build the decomposition using the block structure.
    # Indexing:
    # A is 4x4, indexed as A[row, col] where row = 2*row_block + row_in_block
    # B is 4x9, indexed as B[row, col] where row = 2*row_block + row_in_block, 
    #   col = 3*col_block + col_in_block
    # C is 4x9, indexed as C[row, col] similarly
    
    # For the decomposition, we need:
    # U: (4*4) x rank = 16 x rank
    # V: (4*9) x rank = 36 x rank  
    # W: (9*4) x rank = 36 x rank
    
    # Each "multiplication" in the block decomposition corresponds to one rank-1 term.
    # For a (2x2)*(2x3) product at position (row_block_a, col_block_b) contributing to
    # output block (row_block_c, col_block_c), we have 6 scalar multiplications.
    
    # Let me enumerate all the (2x2)*(2x3) subproblems:
    # For output block (rb_c, cb_c) where rb_c in {0,1}, cb_c in {0,1,2}:
    #   C[rb_c] = A[rb_c, 0] * B[0, cb_c] + A[rb_c, 1] * B[1, cb_c]
    #   where A[rb_c, rb_a] is a 2x2 block, B[rb_b, cb_c] is a 2x3 block
    
    # For each such term, we do a (2x2)*(2x3) schoolbook with 6 multiplications.
    
    rank_terms = []  # List of (U_vec, V_vec, W_vec) for each rank-1 term
    
    for rb_c in range(2):  # output row block
        for cb_c in range(3):  # output col block
            for rb_a in range(2):  # A row block (input row)
                for rb_b in range(2):  # B row block (input row)
                    # We're computing A[rb_c, rb_a] * B[rb_b, cb_c]
                    # This is a (2x2) * (2x3) product.
                    # Schoolbook: for each (i, j, k) where i in {0,1} (row of result),
                    # j in {0,1} (col of A / row of B), k in {0,1,2} (col of result)
                    # C[i,k] += A[i,j] * B[j,k]
                    
                    for i in range(2):  # row within block
                        for j in range(2):  # col of A block / row of B block
                            for k in range(3):  # col within block
                                # This is one scalar multiplication
                                # U index: global (row_a, col_a) = (2*rb_c + i, 2*rb_a + j)
                                # V index: global (row_b, col_b) = (2*rb_b + j, 3*cb_c + k)
                                # W index: global (row_c, col_c) = (3*cb_c + k, 2*rb_c + i)
                                
                                u_idx = (2 * rb_c + i) * 4 + (2 * rb_a + j)
                                v_idx = (2 * rb_b + j) * 9 + (3 * cb_c + k)
                                w_idx = (3 * cb_c + k) * 4 + (2 * rb_c + i)
                                
                                rank_terms.append((u_idx, v_idx, w_idx))
    
    rank = len(rank_terms)
    print(f"Block decomposition rank: {rank}")
    
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    for r, (u_idx, v_idx, w_idx) in enumerate(rank_terms):
        U[u_idx, r] = 1.0
        V[v_idx, r] = 1.0
        W[w_idx, r] = 1.0
    
    # Verify
    valid = _verify_decomposition(U, V, W, 4, 4, 9)
    if not valid:
        print("Block decomposition verification failed!")
        return None
    
    elapsed = time.time() - start_time
    print(f"Block decomposition constructed and verified in {elapsed:.2f}s")
    
    return U, V, W

def _try_linear_dependency_reduction(U, V, W, n, m, p, time_limit=50):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    If some columns of U, V, W are linearly dependent, we can reduce the rank.
    """
    start_time = time.time()
    rank = U.shape[1]
    
    # Check if we can find a lower-rank equivalent decomposition.
    # This is hard in general, but for our structured decomposition, 
    # we might be able to find redundancies.
    
    # For now, just return the same decomposition.
    # A more sophisticated approach would use SVD or rank-revealing QR.
    
    elapsed = time.time() - start_time
    print(f"Linear dependency check took {elapsed:.2f}s")
    
    return U, V, W

def construct():
    """
    Returns a dict with keys 'U', 'V', 'W' representing the tensor decomposition.
    U: (n*m) x rank
    V: (m*p) x rank  
    W: (p*n) x rank
    
    Convention: W[k*n+i, r] corresponds to output element (k, i) of the product.
    For A (n x m) * B (m x p) = C (n x p), the tensor T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    is decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try block decomposition first (should give rank 72)
    result = _try_block_decomposition(n, m, p, time_limit=100)
    
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        print(f"Using block decomposition with rank {rank}")
        
        # Convert to lists with rational entries
        U_list = [[float(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
        V_list = [[float(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
        W_list = [[float(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
        
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed:.2f}s")
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    print("Falling back to schoolbook decomposition")
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = U.shape[1]
    
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
    
    elapsed = time.time() - start_time
    print(f"Total time (schoolbook): {elapsed:.2f}s")
    
    return {"U": U_list, "V": V_list, "W": W_list}
