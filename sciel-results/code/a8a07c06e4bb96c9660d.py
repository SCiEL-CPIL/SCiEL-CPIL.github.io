"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _column_signature(U, V, W, col, n, m, p):
    """Get the set of (i,j,k) triples covered by column col."""
    sig = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if U[i * m + j, col] * V[j * p + k, col] * W[k * n + i, col] != 0:
                    sig.add((i, j, k))
    return sig


def _try_structured_decomposition(n, m, p, deadline):
    """
    Try structured decompositions exploiting the 8=2^3 structure.
    
    Key idea: split k into bits k = 4*a + 2*b + c where a,b,c in {0,1}.
    Try to share U rows across different k values.
    
    Also try: for each j, the V rows V[j*p+k, :] for k=0..7 form a pattern.
    If we can make V[j*p+k, c] depend only on some function of k,
    we might reduce the effective rank.
    """
    start = time.time()
    
    # Approach 1: Try rank = 36 with a structured pattern
    # Split 8 columns of B into two groups of 4
    # C[:, 0:4] = A * B[:, 0:4], C[:, 4:8] = A * B[:, 4:8]
    # If we use the same U structure for both, we save rank.
    
    # For 3x3 x 3x4: schoolbook rank = 36
    # Can we do better? Lower bound = max(9, 12, 12) = 12
    
    # Let's try to find a rank-r decomposition for 3x3 x 3x8 directly
    # using a greedy approach with backtracking
    
    # Start with schoolbook and try to reduce
    U, V, W = _build_schoolbook(n, m, p)
    current_rank = U.shape[1]
    
    # Precompute signatures for all columns
    def compute_all_signatures(U_mat, V_mat, W_mat, rank, n, m, p):
        sigs = []
        for c in range(rank):
            sigs.append(_column_signature(U_mat, V_mat, W_mat, c, n, m, p))
        return sigs
    
    # Try to find a minimal set cover-like structure
    # But this is NP-hard in general, so use heuristics
    
    # Alternative: use the fact that for binary matrices, the problem is
    # equivalent to finding a 3-partite 3-uniform hypergraph with specific properties
    
    # Let's try a direct search for small ranks with structured ansatz
    # Ansatz: U[i,j,c] depends only on (i mod 2, j mod 2, c mod something)
    
    # Try rank = 24 (the lower bound from m*p = p*n = 24)
    # This would require each column to cover exactly 3 triples on average
    
    # For rank = 24, we need 24 columns each covering 3 triples
    # Total triples = 72, so 72/24 = 3 per column
    
    # Try a cyclic construction
    # Label triples as (i, j, k) where i in {0,1,2}, j in {0,1,2}, k in {0,...,7}
    # Column c covers triples where (i, j, k) satisfies some condition
    
    # Cyclic group Z_3 x Z_8 has 24 elements
    # If we can assign each triple to exactly one column via a group action...
    
    # For column c = (a, b) in Z_3 x Z_8:
    #   covers (i, j, k) if i = j + a (mod 3) and k = j*3 + b (mod 8)? 
    #   That doesn't quite work since j*3 mod 8 has limited range.
    
    # Let me try a different approach: use finite field or direct construction
    
    # Actually, let's just try random search with the constraint that
    # entries are in {0, 1} and use a local search to minimize rank
    
    # Start from schoolbook and try to merge columns
    # Two columns c1, c2 can be "merged" if we can find a new column c
    # such that the union of their covered triples is covered by c,
    # and no other triple is covered by c.
    
    # But a single binary column can only cover triples where all three
    # entries are 1. So the covered set of a column is determined by
    # the intersection of the supports of U[:,c], V[:,c], W[:,c].
    
    # This is very restrictive. Let's instead try to find a completely
    # different assignment from scratch.
    
    # For rank r, we have r columns. Each column c defines:
    #   I_c = {(i,j) : U[i*m+j, c] = 1}  (subset of 9 pairs)
    #   J_c = {(j,k) : V[j*p+k, c] = 1}  (subset of 24 pairs)
    #   K_c = {(k,i) : W[k*n+i, c] = 1}  (subset of 24 pairs)
    #   Covered_c = {(i,j,k) : (i,j) in I_c, (j,k) in J_c, (k,i) in K_c}
    
    # We need: for each (i,j,k), exactly one c has (i,j,k) in Covered_c
    
    # This is a very constrained problem. Let's try small r with backtracking.
    
    # For r = 24, each column must cover exactly 3 triples (72/24 = 3)
    # Let's try to construct this
    
    # Use the structure: for each (i,j), there are 8 k values.
    # U[i*m+j, c] is shared across all k for a given (i,j) and column c.
    # So if U[i*m+j, c] = 1, then for all k where V[j*p+k, c] = 1 and W[k*n+i, c] = 1,
    # the triple (i,j,k) is covered.
    
    # This means the coverage is highly structured.
    
    # Let's try: for each column c, pick a subset S_c of the 9 (i,j) pairs.
    # Then V and W must be consistent.
    
    # This is getting complex. Let's just do a bounded random search.
    
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = 24
    # But also, for each i, sum_j sum_k ... constraints give more bounds
    
    # Let's try a few specific ranks
    for r in [24, 27, 30, 32, 36, 40, 48, 54, 60, 64, 68, 72]:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
            
        # Try random binary assignments
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + r * 1000 + attempt)
            U_rand = np.zeros((n * m, r))
            V_rand = np.zeros((m * p, r))
            W_rand = np.zeros((p * n, r))
            
            # Random binary with varying density
            density = 0.2 + 0.3 * (attempt % 10) / 10
            U_rand = (np.random.rand(n * m, r) < density).astype(float)
            V_rand = (np.random.rand(m * p, r) < density).astype(float)
            W_rand = (np.random.rand(p * n, r) < density).astype(float)
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            if U_rand[i * m + j, c] and V_rand[j * p + k, c] and W_rand[k * n + i, c]:
                                s += 1
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break  # Found a valid decomposition for this rank
    
    if best_rank < n * m * p:
        return best_rank, best_U, best_V, best_W
    
    return None


def _try_greedy_merge(n, m, p, deadline):
    """
    Start from schoolbook and try to merge columns greedily.
    Two columns can be merged if their covered triples are disjoint
    and we can create a new column that covers the union.
    
    But as noted, a single binary column can only cover triples where
    all three factors are 1. So merging is very restrictive.
    
    Instead, try to find a different binary assignment with fewer columns.
    """
    start = time.time()
    
    # Use a local search: start from schoolbook, randomly perturb,
    # and try to reduce the number of columns while maintaining validity.
    
    # This is a hard problem. Let's try a simpler approach:
    # Use the structure that 8 = 2^3 and try to exploit bit-level sharing.
    
    # For k in {0,...,7}, write k in binary: k = 4*b2 + 2*b1 + b0
    # Try to make V[j*p+k, c] depend on b0, b1, b2 separately.
    
    # If V[j*p+k, c] = v0[j, c, b0] * v1[j, c, b1] * v2[j, c, b2]
    # and similarly for W, we might factor the rank.
    
    # But this is speculative. Let's just try a direct numerical search
    # with a tighter time budget.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard: must finish within 120s, use 110s as safety
    deadline = time.time() + 110
    
    # First, get the schoolbook solution as fallback
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try structured decomposition
    result = _try_structured_decomposition(n, m, p, deadline)
    if result is not None and result[0] < best_rank:
        r, U2, V2, W2 = result
        best_rank = r
        best_U, best_V, best_W = U2, V2, W2
    
    # If we have time, try greedy merge
    if time.time() < deadline - 10 and best_rank == schoolbook_rank:
        result2 = _try_greedy_merge(n, m, p, deadline)
        if result2 is not None and result2[0] < best_rank:
            r, U2, V2, W2 = result2
            best_rank = r
            best_U, best_V, best_W = U2, V2, W2
    
    # Convert to lists and round to half-integers
    U_list = _round_to_half(best_U).tolist()
    V_list = _round_to_half(best_V).tolist()
    W_list = _round_to_half(best_W).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
