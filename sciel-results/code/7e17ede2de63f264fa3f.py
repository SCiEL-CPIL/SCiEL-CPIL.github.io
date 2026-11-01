"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition.

We try to find a decomposition with rank below the schoolbook rank of 50.
For small dimensions like 2x5x5, the true rank is unknown but likely below 50.

Strategy:
- Start with schoolbook rank 50 as a guaranteed baseline.
- Use integer/half-integer entries (0, 0.5, 1, -0.5, -1) to keep exactness.
- Attempt to find a lower-rank decomposition via a structured approach:
  We exploit the fact that C = A * B where A is 2x5 and B is 5x5.
  The output C is 2x5. Each entry C[i][k] = sum_j A[i][j] * B[j][k].
  
  A bilinear decomposition of rank r means we can write:
  C[i][k] = sum_{r=1}^{R} u_r[i,j] * A[i][j] * v_r[j,k] * B[j,k] * w_r[k,i]
  
  where u_r, v_r, w_r are fixed tensors (the factor matrices).

  For the schoolbook, each term isolates one (i,j,k) triple.
  
  To reduce rank, we need to combine terms. One natural approach:
  For a fixed i, the row i of C is a linear combination of rows of B weighted by A[i].
  So C[i] = A[i] @ B (as a row vector).
  
  This means for each output row i, we need to compute A[i] @ B.
  If we could compute A[i] @ B using a rank-r_i bilinear algorithm for 1x5x5 matrix mult,
  then total rank would be sum of r_i for i=0,1.
  
  The rank of 1x5x5 matrix mult: C[k] = sum_j A[j] * B[j][k].
  This is just a vector-matrix product. The tensor is essentially the "identity" in j
  coupled with the row of B. Actually, this is just: for each k, C[k] = sum_j A[j] B[j,k].
  
  The bilinear complexity of 1xm x mxp is mp (schoolbook). Can we do better?
  For 1x5x5, the rank is 25 (each C[k] depends on all j). Actually no:
  C[k] = sum_j A[j] * B[j,k] for k=0..4.
  This is 5 separate dot products of length 5. The total number of bilinear terms is 25.
  But can we share computations? Since A[j] is shared across k, we might factor.
  
  Actually, the tensor for 1xm x mxp is: T_{j,k,l} where l is the output index (0..p-1),
  and the input A has m components, B has m*p components.
  T = sum_{j,k} A[j] * B[j,k] * e_k (output basis).
  This tensor has rank mp because each output coordinate k requires summing over j,
  and there's no sharing between different k's in a bilinear decomposition... 
  unless we use Strassen-like tricks. But for 1xm x mxp, the rank is exactly mp.
  
  So total rank would be 2*25 = 50, which is the schoolbook. No improvement.

  However, we might be able to do better by sharing between the two rows of A.
  Let A have rows a0 and a1. C[0] = a0 @ B, C[1] = a1 @ B.
  
  We can write:
  C[0] = a0 @ B
  C[1] = a1 @ B
  
  Consider the 2x5 matrix A. We need to compute A @ B where B is 5x5.
  
  Another approach: view B as a 5x5 matrix. We can use a rank-r decomposition for 5x5x5
  matrix multiplication, but we only use part of it. Actually, we need A (2x5) times B (5x5).
  
  Let's try a direct search with a small rank, using a genetic algorithm or local search
  over integer/half-integer entries.

  Given the time constraint, I'll implement a local search that starts from schoolbook
  and tries to merge terms.

  Actually, let me try a different approach: use the known result that for 2xnxn,
  the rank might be lower. Let me try rank 30, 35, 40, 45 and see if we can find
  a valid decomposition via local search.
"""
import time
import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]
N, M, P = 2, 5, 5
NM = N * M  # 10
MP = M * P  # 25
PN = P * N  # 10

def _schoolbook(n, m, p):
    """Guaranteed-exact schoolbook decomposition of rank n*m*p."""
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


def _verify_exact(U, V, W, n, m, p, rng, num_trials=16):
    """Check the tensor identity exactly on random integer bases."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    for _ in range(num_trials):
        A = rng.integers(-4, 5, size=(n, m))
        B = rng.integers(-4, 5, size=(m, p))
        C_true = A @ B
        C_rec = np.zeros((n, p), dtype=np.float64)
        for r in range(rank):
            u_r = Ua[:, r].reshape(n, m)
            v_r = Va[:, r].reshape(m, p)
            w_r = Wa[:, r].reshape(p, n)
            a_part = u_r * A
            b_part = v_r * B
            M = a_part @ b_part
            C_rec += M * w_r.T
        if not np.allclose(C_true, C_rec, atol=1e-9):
            return False
    return True


def _try_local_search(n, m, p, target_rank, max_iters=2000, rng=None):
    """Attempt to find a rank=target_rank decomposition via local search."""
    if rng is None:
        rng = random.Random(42)
    
    # Start with a random initialization using values in {-1, 0, 1}
    U = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
    V = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
    W = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
    
    def _error(U, V, W, A, B):
        """Compute the error of the decomposition for given A, B."""
        C_true = A @ B
        C_rec = np.zeros((n, p), dtype=np.float64)
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for r in range(target_rank):
            u_r = Ua[:, r].reshape(n, m)
            v_r = Va[:, r].reshape(m, p)
            w_r = Wa[:, r].reshape(p, n)
            a_part = u_r * A
            b_part = v_r * B
            M = a_part @ b_part
            C_rec += M * w_r.T
        return np.sum((C_true - C_rec) ** 2)
    
    # Test on a fixed set of random matrices
    test_rng = np.random.default_rng(12345)
    test_mats = []
    for _ in range(5):
        A = test_rng.integers(-3, 4, size=(n, m))
        B = test_rng.integers(-3, 4, size=(m, p))
        test_mats.append((A, B))
    
    def _total_error(U, V, W):
        return sum(_error(U, V, W, A, B) for A, B in test_mats)
    
    current_U, current_V, current_W = U, V, W
    current_err = _total_error(current_U, current_V, current_W)
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_err = current_err
    
    values = [-1, -0.5, 0, 0.5, 1]
    
    for it in range(max_iters):
        # Random mutation: change one entry in one of the matrices
        mat_idx = rng.choice([0, 1, 2])
        if mat_idx == 0:
            row = rng.randrange(n*m)
            col = rng.randrange(target_rank)
            old = U[row][col]
            new = rng.choice([v for v in values if v != old])
            U[row][col] = new
            err = _total_error(U, V, W)
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                U[row][col] = old
        elif mat_idx == 1:
            row = rng.randrange(m*p)
            col = rng.randrange(target_rank)
            old = V[row][col]
            new = rng.choice([v for v in values if v != old])
            V[row][col] = new
            err = _total_error(U, V, W)
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                V[row][col] = old
        else:
            row = rng.randrange(p*n)
            col = rng.randrange(target_rank)
            old = W[row][col]
            new = rng.choice([v for v in values if v != old])
            W[row][col] = new
            err = _total_error(U, V, W)
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                W[row][col] = old
    
    if best_err < 1e-6:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 100.0  # safety margin before 150s limit
    
    rng = np.random.default_rng(2024)
    
    # Baseline: guaranteed-exact schoolbook decomposition (rank 50).
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Try ranks from 45 down to 30
    for target_rank in [45, 40, 35, 30]:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            candidate = _try_local_search(n, m, p, target_rank, 
                                          max_iters=500, 
                                          rng=random.Random(42 + attempt))
            if candidate is not None:
                # Verify with more trials
                if _verify_exact(candidate["U"], candidate["V"], candidate["W"], 
                                n, m, p, rng, num_trials=32):
                    best_rank = target_rank
                    result = candidate
                    break  # Found a valid decomposition at this rank, try lower
    
    # Final verification
    if not _verify_exact(result["U"], result["V"], result["W"], n, m, p, rng, num_trials=16):
        # Fall back to schoolbook if verification fails
        result = _schoolbook(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
