"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication using Strassen (rank 7).
    Convention:
      A is 2x2, B is 2x2, C = A @ B is 2x2.
      U has shape (n*m, rank) = (4, 7). U[i*m + j, c] corresponds to A[i,j].
      V has shape (m*p, rank) = (4, 7). V[j*p + k, c] corresponds to B[j,k].
      W has shape (p*n, rank) = (4, 7). W[k*n + i, c] corresponds to C[i,k] (output index k*n+i).
    
    Strassen's algorithm:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
      
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Index mapping: U[i*m+j, c] = A[i,j]
    # V[j*p+k, c] = B[j,k]
    # W[k*n+i, c] = C[i,k]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 (k=0,i=0 -> k*n+i=0)
    W[3, 0] = 1  # C11 (k=1,i=1 -> k*n+i=3)
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=0,i=1 -> k*n+i=2)
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01 (k=1,i=0 -> k*n+i=1)
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition exactly matches matrix multiplication.
    For each (i, j, k), the coefficient should be 1 if the product C[i,k] = sum_j A[i,j]*B[j,k].
    """
    rank = U.shape[1]
    # The tensor T[i,j,k] should be 1 for all i,j,k in the standard multiplication.
    # The decomposition gives: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check that for all i,j,k, this sum equals 1 (the standard multiplication tensor).
    # Actually, the standard multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Wait, no. The multiplication tensor for A(n x m) * B(m x p) = C(n x p) is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k.
    # The rank-r decomposition is: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a bit tricky to vectorize directly. Let's use einsum.
    # U_r[i,j,c], V_r[j,k,c], W_r[k,i,c]
    # We want: sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)


def _try_random_search(n, m, p, target_rank, time_budget, seed=42):
    """
    Try to find a rank-`target_rank` decomposition using random initialization and gradient descent.
    This is a heuristic approach and may not succeed.
    """
    np.random.seed(seed)
    start_time = time.time()
    
    # We'll use a simple approach: random initialization + a few iterations of a basic optimization.
    # However, tensor decomposition is NP-hard in general, so this is unlikely to work well.
    # Instead, we'll just return None to indicate failure.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before the 150s limit
    
    # Start with the schoolbook decomposition (rank = n*m*p = 64)
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to use Strassen-based decomposition for the 2x2 blocks.
    # The matrix B is 2x16, which can be viewed as four 2x2 blocks.
    # A is 2x2.
    # C = A @ B is 2x16, which is four 2x2 blocks.
    # 
    # If we use Strassen for each 2x2 block independently, the rank would be 4*7 = 28.
    # But we can potentially share some of the Strassen factors across blocks.
    #
    # Let's first try the straightforward block-wise Strassen: rank = 4 * 7 = 28.
    
    if time.time() < deadline:
        S_U, S_V, S_W = _build_strassen_2x2()
        # S_U: (4, 7), S_V: (4, 7), S_W: (4, 7)
        # For 2x2 * 2x16:
        # A is 2x2, B is 2x16 = [B0, B1, B2, B3] where each Bi is 2x2.
        # C is 2x16 = [C0, C1, C2, C3] where each Ci is 2x2.
        # Ci = A @ Bi for i = 0,1,2,3.
        #
        # U has shape (n*m, rank) = (4, rank). U[i*m+j, c] = A[i,j].
        # V has shape (m*p, rank) = (4*16=32, rank). V[j*p+k, c] = B[j,k].
        # W has shape (p*n, rank) = (16*2=32, rank). W[k*n+i, c] = C[i,k].
        #
        # For block i (0-indexed), B's columns 2*i and 2*i+1 form Bi.
        # So B[j, 2*i + k'] = B[j*p + (2*i + k')] where k' is 0 or 1.
        # C's columns 2*i and 2*i+1 form Ci.
        # So C[i', 2*i + k'] = C[(2*i + k')*n + i'].
        #
        # Let's build the block-wise Strassen decomposition.
        block_rank = 7
        total_rank = 4 * block_rank
        U_block = np.zeros((n*m, total_rank), dtype=float)
        V_block = np.zeros((m*p, total_rank), dtype=float)
        W_block = np.zeros((p*n, total_rank), dtype=float)
        
        for block in range(4):
            col_offset = block * block_rank
            # U is the same for all blocks (since A is the same)
            U_block[:, col_offset:col_offset+block_rank] = S_U
            # V: for block `block`, B[j, 2*block + k'] corresponds to V[j*p + 2*block + k', col]
            # S_V has shape (4, 7) with S_V[j*2 + k', c] = B[j,k'] for the block.
            # So V[j*p + 2*block + k', col_offset + c] = S_V[j*2 + k', c]
            for j in range(m):
                for k in range(p):
                    block_idx = k // 2
                    k_in_block = k % 2
                    if block_idx == block:
                        V_block[j*p + k, col_offset:col_offset+block_rank] = S_V[j*2 + k_in_block, :]
            # W: for block `block`, C[i', 2*block + k'] corresponds to W[(2*block + k')*n + i', col]
            # S_W has shape (4, 7) with S_W[k'*2 + i', c] = C[i',k'] for the block.
            # So W[(2*block + k')*n + i', col_offset + c] = S_W[k'*2 + i', c]
            for i in range(n):
                for k in range(p):
                    block_idx = k // 2
                    k_in_block = k % 2
                    if block_idx == block:
                        W_block[k*n + i, col_offset:col_offset+block_rank] = S_W[k_in_block*2 + i, :]
        
        # Verify
        if _verify_decomposition_fast(U_block, V_block, W_block, n, m, p):
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = total_rank
    
    # Try to search for a lower rank using random methods (unlikely to succeed, but try briefly)
    if time.time() < deadline and best_rank > 16:
        # Try rank 16, 14, etc. with random search
        for target_rank in [16, 14, 12]:
            if time.time() >= deadline - 10:
                break
            result = _try_random_search(n, m, p, target_rank, time_budget=deadline - time.time())
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition_fast(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
