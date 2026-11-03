"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction as Fr

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    rank = len(W[0])
    # Build tensor: T[a, b, c] = sum_k U[a,k]*V[b,k]*W[c,k]
    # where a = i*m+j, b = j*p+k, c = k'*n+i
    # We need T to equal the standard tensor: T[i*m+j, j*p+k, k*n+i] = 1 iff (i,j,k) match
    # i.e. T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # For standard matmul: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T_{i,j,k} where output index is (i,k)
    # The factor matrices are:
    #   U indexed by (i,j) -> rank
    #   V indexed by (j,k) -> rank
    #   W indexed by (k,i) -> rank
    # So T[i*m+j, j*p+k, k*n+i] should be 1 for all i,j,k
    # And 0 otherwise
    
    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = Fr(0)
                for r in range(rank):
                    val += Fr(U[a][r]) * Fr(V[b][r]) * Fr(W[c][r])
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook = _build_schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition using randomized linear algebra
    # Strategy: The tensor T lives in (R^{n*m}) * (R^{m*p}) * (R^{p*n})
    # A rank-r decomposition means T can be written as sum of r rank-1 tensors
    # Each rank-1 tensor is determined by vectors u in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # such that T[a,b,c] = sum_r u[a,r] * v[b,r] * w[c,r]
    
    # We can set up: for each fixed c, define T_c[a,b] = T[a,b,c]
    # Then T_c = U_c * V_c^T where U_c has shape (n*m, r) and V_c has shape (m*p, r)
    # Actually T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # So for fixed c: T[a,b,c] = sum_r (U[a,r] * W[c,r]) * V[b,r]
    # Let M_c[a,r] = U[a,r] * W[c,r], then T_c = M_c * V^T
    
    # This is a constrained factorization problem.
    # 
    # Alternative approach: iterate over possible ranks from lower bound to schoolbook
    # and try to solve the bilinear system.
    
    # The matrix multiplication tensor for [n,m,p] has rank at least max(n*m, m*p, p*n)
    # For [4,5,5]: max(20, 25, 20) = 25
    # Lower bound is also n*m*p / min(n,m,p) in some cases, but the trivial lower bound is max(n*m, m*p, p*n)
    
    min_rank = max(n * m, m * p, p * n)
    max_rank = n * m * p
    
    # For [4,5,5], min_rank = 25, max_rank = 100
    # Trying to find rank between 25 and 99 is computationally hard.
    # Let's try a few ranks near the lower bound with randomized initialization.
    
    best_result = schoolbook
    best_rank = max_rank
    
    # Try ranks from min_rank upward, but limit search time
    # For small ranks, we can use a greedy/iterative approach
    # For larger ranges, this becomes intractable
    
    # Given the time constraint, let's just return the schoolbook decomposition
    # which is guaranteed to be valid and has rank 100.
    
    # Actually, let me try one specific known result:
    # For [2,2,2], rank is 7 (Strassen)
    # For [4,4,4], rank is at most 49 (using Strassen recursively)
    # For [4,5,5], we could potentially use a [4,4,4] decomposition plus extra terms
    
    # Let's try: use Strassen for the 4x4x4 part and add 4*5 + 4*5 + 5*5 - 4*4*4 extra terms?
    # This is getting complex. The safe approach is to return schoolbook.
    
    # But let me try one more thing: a simple random search for low rank
    # This is unlikely to succeed but might find something
    
    if time.time() < deadline - 10:
        # Try rank = min_rank + a few
        for test_rank in range(min_rank, min(min_rank + 10, max_rank)):
            if time.time() > deadline - 5:
                break
            # Random initialization
            random.seed(42)
            U_rand = [[random.randint(-2, 2) for _ in range(test_rank)] for _ in range(n * m)]
            V_rand = [[random.randint(-2, 2) for _ in range(test_rank)] for _ in range(m * p)]
            W_rand = [[random.randint(-2, 2) for _ in range(test_rank)] for _ in range(p * n)]
            
            if _validate(U_rand, V_rand, W_rand, n, m, p):
                best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                best_rank = test_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
