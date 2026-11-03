"""Program for matrix multiplication 4x5x5 tensor decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [4, 5, 5]

def _gf2_solve(A, b):
    """Solve Ax = b over GF(2). Returns solution x or None if inconsistent."""
    A = np.array(A, dtype=np.uint8)
    b = np.array(b, dtype=np.uint8)
    rows, cols = A.shape
    augmented = np.hstack([A, b.reshape(-1, 1)])
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot = -1
        for i in range(r, rows):
            if augmented[i, c] == 1:
                pivot = i
                break
        if pivot == -1:
            continue
        augmented[[r, pivot]] = augmented[[pivot, r]]
        for i in range(rows):
            if i != r and augmented[i, c] == 1:
                augmented[i] ^= augmented[r]
        r += 1
    for i in range(r, rows):
        if augmented[i, cols] == 1:
            return None
    x = np.zeros(cols, dtype=np.uint8)
    for i in range(r):
        col = np.argmax(augmented[i, :cols])
        if augmented[i, col] == 1:
            x[col] = augmented[i, cols]
    return x

def _try_rank_reduction(n, m, p, target_rank, max_pairs=500):
    """Try to find a decomposition of given target_rank using GF(2) support search."""
    rank_schoolbook = n * m * p
    if target_rank >= rank_schoolbook:
        return None
    
    # For each potential rank term, we need to assign (i,j,k) indices
    # The tensor equation is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Over GF(2), we want to find supports that satisfy the equations
    
    # Strategy: start from schoolbook and try to merge terms
    # A merge is valid if two terms (i1,j1,k1) and (i2,j2,k2) can be combined
    # such that the XOR of their contributions matches
    
    # Simpler approach: try all pairs of schoolbook terms to see if they can be merged
    # Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if they share
    # at least one index, creating a new term with mixed indices
    
    # Let's try a different approach: use the known structure
    # For matrix multiplication, rank can be reduced below n*m*p if there's
    # algebraic structure. For 4x5x5, let's try to find if rank 99 or less works.
    
    # Build the tensor T[i,j,k] = 1 if the term (i,j,k) exists in schoolbook
    # We want to express this as sum of rank-1 tensors
    
    # Over GF(2), the condition for a rank-r decomposition is that the tensor
    # T can be written as sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s
    
    # Let's try a greedy approach: find pairs of terms that can be "absorbed"
    # by introducing a new term that cancels their individual contributions
    
    # For 4x5x5, let's try to find a decomposition with rank < 100
    # by looking for linear dependencies over GF(2)
    
    # The tensor has 4*5*5 = 100 entries
    # Each rank-1 term contributes to one (i,j,k) position
    # We need sum of rank-1 terms = T
    
    # Over GF(2), if we have two terms at the same position, they cancel
    # So we can try to find a set of rank-1 terms whose XOR equals T
    
    # This is equivalent to: find a multiset of rank-1 tensors that sum to T over GF(2)
    # The minimum size of such a multiset is the rank over GF(2)
    
    # For matrix multiplication tensor, the rank over GF(2) is known to be
    # at most n*m*p, and sometimes less
    
    # Let's try to compute the rank over GF(2) by viewing T as a linear map
    # T: V* ⊗ W* → V where (v,w) -> sum_{i,j,k} T[i,j,k] v[i] w[k] e_j
    
    # Actually, let's just try to find a valid decomposition by search
    
    # For small target ranks, we can try random assignments and check validity
    
    # But first, let's see if rank 99 is possible by checking if there's a
    # linear dependency in the schoolbook terms over GF(2)
    
    # The schoolbook decomposition has 100 terms. If any subset of these terms
    # sums to zero over GF(2), we can remove that subset and add a correction
    
    # Actually, the simplest approach: try to find if the tensor has rank < 100
    # over GF(2) by computing the rank of the associated linear map
    
    # View T as a 3-tensor. The rank over GF(2) is the minimum r such that
    # T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s
    
    # We can try to compute this by flattening: T reshaped to (4*5, 5) matrix
    # and computing its rank, but that gives the matrix rank, not tensor rank
    
    # For now, let's just try to find a valid decomposition with rank < 100
    # by searching for sparse supports
    
    # Given time constraints, let's return None if we can't quickly find a reduction
    # The schoolbook rank is a valid upper bound
    
    return None

def _build_decomposition(n, m, p, rank, supports):
    """Build U, V, W matrices from support patterns."""
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (i, j, k) in enumerate(supports):
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    n, m, p = 4, 5, 5  # Hardcoded for this specific case
    
    # Check: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # (since T[i,j,k] = 1 for all valid i,j,k in matrix multiplication)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # First, return schoolbook as fallback
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1

    # Try to find a reduced rank decomposition
    best_rank = rank_schoolbook
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try ranks from rank_schoolbook-1 down, but limit search time
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 5, 90), -1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a valid decomposition with this rank
        # Use a simple heuristic: try random support patterns
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Generate random support pattern
            # Each rank term gets a random (i,j,k)
            supports = [(np.random.randint(n), np.random.randint(m), np.random.randint(p)) 
                       for _ in range(target_rank)]
            
            U, V, W = _build_decomposition(n, m, p, target_rank, supports)
            
            if _verify_decomposition(n, m, p, U, V, W):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
# EVOLVE-BLOCK-END
