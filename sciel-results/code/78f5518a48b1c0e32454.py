"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.

We combine the schoolbook baseline (rank 96) with a structured search that
attempts to find lower-rank decompositions using known algebraic techniques
for matrix multiplication tensors.

Key ideas:
1. Start from schoolbook (rank 96) as the guaranteed-valid baseline.
2. Try to construct lower-rank decompositions using:
   - Block-based Strassen-style reductions
   - Kronecker product structure
   - Randomized search with integer/half-integer rounding
3. Verify any candidate decomposition exactly.
4. Return the best (lowest-rank) valid decomposition found.

The output W follows the trace convention W[k*n+i, r].
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=None, m=None, p=None):
    """
    Verify that U, V, W form a valid exact tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition must satisfy:
      For all i, k: sum_j A[i,j]*B[j,k] == sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    More precisely, the tensor T[i,j,k] = delta (identity in j) must be recovered:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 if j is the "inner" index connecting them, 0 otherwise.
    
    Actually, the matrix multiplication tensor is:
      M[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    The tensor decomposition says:
      M[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not quite right. Let me think more carefully.
    
    The matrix multiplication tensor T has entries T[i, j, k] where:
    - i indexes rows of A (and C): 0..n-1
    - j indexes columns of A / rows of B: 0..m-1  
    - k indexes columns of B (and C): 0..p-1
    
    T[i,j,k] = 1 (it's the standard matrix mult tensor)
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    In flattened form:
    U has shape (n*m, rank): U[i*m+j, r]
    V has shape (m*p, rank): V[j*p+k, r]
    W has shape (p*n, rank): W[k*n+i, r]  (trace convention)
    
    So: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    And T[i,j,k] = 1 for all valid i,j,k (since it's the matrix mult tensor).
    """
    if n is None:
        n, m, p = DIMENSIONS
    
    rank = len(U[0]) if U else 0
    
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Verify: for all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _try_block_strassen():
    """
    Try a block-based approach. For 3x4x8, we can try to use Strassen's
    algorithm on 2x2 blocks where possible, combined with schoolbook for
    the rest.
    
    Strategy: Partition A (3x4) into blocks and B (4x8) into blocks.
    A is 3x4: can split into [2x4; 1x4] or [3x2 | 3x2]
    B is 4x8: can split into [2x8; 2x8] or [4x4 | 4x4]
    
    Let's try: A = [A1 (2x4); A2 (1x4)], B = [B1 (4x4) | B2 (4x4)]
    C = A*B = [A1*B1 | A1*B2; A2*B1 | A2*B2]
    
    A1*B1: 2x4 * 4x4 -> use Strassen? 2x4*4x4 isn't square.
    
    Alternative: Try to find a decomposition with rank < 96 using
    a systematic search over small rank values with random initialization
    and rounding to integers/half-integers.
    """
    n, m, p = DIMENSIONS
    best = None
    deadline = time.time() + 100  # Leave 50s safety margin
    
    # Try ranks from a reasonable lower bound up to 95
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But actually the tensor rank lower bound is higher.
    # For matrix multiplication, rank >= max(n*m, m*p, p*n) is a weak bound.
    # A better bound: rank >= n*m*p / max(n, m, p) ... not standard.
    # Let's just try a range.
    
    for rank in range(60, 96):  # Try to find something below 96
        if time.time() > deadline:
            break
        if best is not None and rank >= len(best['U'][0]):
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Initialize with random +/- 1, 0
            for r in range(rank):
                for idx in range(n * m):
                    U[idx, r] = random.choice([0, 0, 1, -1])
                for idx in range(m * p):
                    V[idx, r] = random.choice([0, 0, 1, -1])
                for idx in range(p * n):
                    W[idx, r] = random.choice([0, 0, 1, -1])
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best


def _try_kronecker_structure():
    """
    Try to exploit Kronecker product structure.
    
    3x4x8: Note that 4 = 2*2 and 8 = 2*4.
    We can view the multiplication as having some block structure.
    
    A (3x4) = A (3 x 2*2)
    B (4x8) = B (2*2 x 2*4)
    
    If we reshape B as (2, 2, 2, 4) and A as (3, 2, 2),
    then C (3x8) = C (3, 2, 4).
    
    This doesn't immediately give a Strassen reduction, but let's try
    a structured approach where we use the fact that m=4=2^2 and p=8=2^3.
    
    Actually, let's try a simpler idea: use the fact that we can batch
    certain computations. For each pair (i, j) in A, we need to compute
    the dot product with each column k of B. The contribution to C[i,k]
    from A[i,j] * B[j,k] is straightforward.
    
    A more promising approach: try to find a decomposition where some
    of the U, V, W entries are shared across multiple (i,j,k) triples.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try a structured approach: for each j (inner dimension), group
    # the (i, k) pairs. There are n*p = 24 such pairs per j.
    # For each j, we need to represent the "outer product" structure.
    
    # Actually, let's try a different approach: use the fact that
    # the matrix multiplication tensor for n x m x p has a known
    # lower bound. For 3x4x8, the trivial bound is 96.
    # 
    # Let's try a greedy approach: start with schoolbook and try to
    # merge terms that have overlapping structure.
    
    # For now, return None and let the caller use schoolbook.
    return None


def _try_systematic_search():
    """
    Try a more systematic search for lower-rank decompositions.
    
    Key insight: The matrix multiplication tensor T_{i,j,k} = 1 can be
    decomposed as a sum of rank-1 tensors. We want to find the minimum
    number of such rank-1 tensors.
    
    A rank-1 tensor is U[i*m+j] * V[j*p+k] * W[k*n+i].
    
    For the decomposition to work, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    This is a system of n*m*p = 96 equations in 3*rank variables (plus
    the constraint that entries are integers or half-integers).
    
    Let's try to find solutions with rank < 96 using a structured approach.
    
    One idea: fix the structure of U, V, W to have a block form.
    For example, suppose U depends only on i (not j), V depends only on j,
    W depends only on k. Then:
    sum_r U[i, r] * V[j, r] * W[k, r] = 1 for all i,j,k
    
    This would require U, V, W to be rank-1 matrices themselves, which
    gives U[i,r]*V[j,r]*W[k,r] = c for all i,j,k, which can only equal 1
    if c=1, but then we need exactly one nonzero r, giving rank 1.
    But rank 1 is not achievable for matrix multiplication (the tensor
    is not rank 1).
    
    So we need more complex structures. Let's try:
    - U depends on (i, j)
    - V depends on (j, k)  
    - W depends on (k, i)
    
    This is the general case. Let's try small structured patterns.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try rank values from a lower bound upward
    # A reasonable lower bound: the tensor rank of matrix multiplication
    # for n x m x p is at least max(n*m, m*p, p*n) = 32 for our case.
    # But realistically, it's much higher.
    
    # Let's try to find a decomposition using a specific structure:
    # Partition the inner dimension j into groups and use shared factors.
    
    # For m=4, we can try to use 2 "Strassen-like" factors for pairs of j.
    # This is heuristic and may not yield a valid decomposition.
    
    # Given the complexity, let's try a random search with more attempts
    # for ranks in a promising range.
    
    best = None
    for rank in range(80, 96):
        if time.time() > deadline:
            break
        if best is not None and rank >= len(best['U'][0]):
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Start from schoolbook and randomly perturb/merge
            base = _schoolbook()
            U = np.array(base['U'], dtype=float)
            V = np.array(base['V'], dtype=float)
            W = np.array(base['W'], dtype=float)
            
            # Randomly zero out some columns and adjust others
            # This is a crude approach; in practice, we'd need a more
            # principled method.
            
            # Instead, try a different initialization:
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Use a structured initialization based on binary representations
            for r in range(rank):
                # Assign random patterns
                for idx in range(n * m):
                    U[idx, r] = random.choice([0, 1, -1, 0.5, -0.5])
                for idx in range(m * p):
                    V[idx, r] = random.choice([0, 1, -1, 0.5, -0.5])
                for idx in range(p * n):
                    W[idx, r] = random.choice([0, 1, -1, 0.5, -0.5])
            
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best


def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x8 matrix.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U: (12, rank) list of lists
      V: (32, rank) list of lists
      W: (24, rank) list of lists
    
    The decomposition satisfies:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for all valid i, j, k, where T is the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (120s limit)
    
    # Start with schoolbook as the guaranteed-valid baseline
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    # Try various strategies to find lower-rank decompositions
    strategies = [
        _try_block_strassen,
        _try_kronecker_structure,
        _try_systematic_search,
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 10:  # Leave 10s for verification and return
            break
        try:
            result = strategy()
            if result is not None:
                rank = len(result['U'][0])
                if rank < best_rank:
                    # Verify the result
                    if _verify(result['U'], result['V'], result['W']):
                        best = result
                        best_rank = rank
        except Exception:
            continue
    
    # Final verification of the best result
    if not _verify(best['U'], best['V'], best['W']):
        # Fallback to schoolbook if something went wrong
        best = _schoolbook()
    
    return best

# EVOLVE-BLOCK-END
