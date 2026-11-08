"""Matrix multiplication (2,3,9) rank search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    # U:(n*m,r) V:(m*p,r) W:(p*n,r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    # exact check: entries should be exactly 1 (within tiny fp error for halves)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _kron_decomposition():
    """
    Kronecker-product based decomposition.
    Models 2x3 * 3x9 as three independent 2x3 * 3x3 block multiplications.
    Each 2x3*3x3 block has schoolbook rank 18.
    Total rank = 3 * 18 = 54.
    """
    n, m, p = DIMENSIONS
    # p=9 is 3 blocks of size 3
    block_size = 3
    num_blocks = p // block_size  # 3
    
    # For each block b (0,1,2), we multiply A (2x3) by B_b (3x3)
    # The result is C_b (2x3)
    # Schoolbook for each block: rank = 2*3*3 = 18
    # Total rank = 3 * 18 = 54
    
    rank = num_blocks * n * m * block_size  # 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for b in range(num_blocks):
        # Block b handles columns b*block_size to (b+1)*block_size - 1 of the 3x9 matrix
        # and rows b*block_size to (b+1)*block_size - 1 of the 9x? output
        # Wait, the output is 2x9. The block b produces columns b*3 to (b+1)*3-1.
        # The input B_b is rows 0..2 (all 3 rows of A's columns) and cols b*3..(b+1)*3-1 of B.
        # Actually, A is 2x3, B is 3x9.
        # C = A * B. C[i, k] = sum_j A[i,j] * B[j,k]
        # For block b, k ranges from b*3 to (b+1)*3-1.
        # So B_b is the submatrix of B with columns b*3 to (b+1)*3-1.
        # B_b is 3x3.
        # The multiplication is A (2x3) * B_b (3x3) = C_b (2x3).
        # Schoolbook for this: for each i in 0..1, j in 0..2, k' in 0..2 (local k)
        # C_b[i, k'] += A[i,j] * B_b[j, k']
        # Global k = b*3 + k'
        
        for i in range(n):
            for j in range(m):
                for kp in range(block_size):
                    # U[i*m+j, col] = A[i,j]
                    U[i * m + j, col] = 1
                    # V[j*p + (b*block_size + kp), col] = B[j, b*3+kp]
                    V[j * p + (b * block_size + kp), col] = 1
                    # W[(b*block_size + kp)*n + i, col] = C_b[i, kp] which is C[i, b*3+kp]
                    W[(b * block_size + kp) * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin under 150s kill
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try Kronecker decomposition first
    kron_result = _kron_decomposition()
    if _verify(np.array(kron_result['U']), np.array(kron_result['V']), np.array(kron_result['W']), n, m, p, len(kron_result['U'][0])):
        kron_rank = len(kron_result['U'][0])
        if kron_rank < best_rank:
            best = kron_result
            best_rank = kron_rank
    
    # integer / half-integer entry sets
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
    ]

    random.seed(12345)
    low_bound = max(n * m, m * p, p * n)
    rmax = best_rank

    for r in range(low_bound, rmax + 1):
        if time.time() > deadline:
            break
        # number of trials scales; keep modest so each rank gets a chance
        trials = 4000
        for _ in range(trials):
            if time.time() > deadline:
                break
            vs = random.choice(val_sets)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                # sparsity: keep some entries zero to help structure
                for idx in range(n * m):
                    if random.random() < 0.5:
                        U[idx, c] = random.choice(vs)
                for idx in range(m * p):
                    if random.random() < 0.5:
                        V[idx, c] = random.choice(vs)
                for idx in range(p * n):
                    if random.random() < 0.5:
                        W[idx, c] = random.choice(vs)
            if _verify(U, V, W, n, m, p, r):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                # found valid at this rank -> lower bound; cannot beat it unless lower rank,
                # but we are iterating ascending so this is the minimum found.
                return best
        # if r == low_bound and no valid, continue to next rank
    return best
# EVOLVE-BLOCK-END
