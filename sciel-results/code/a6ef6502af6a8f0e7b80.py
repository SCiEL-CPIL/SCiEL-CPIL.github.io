"""Combined program: deterministic algebraic construction with verification fallback."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Target tensor: non-zero at (i*m+j, j*p+k, k*n+i) for all i,j,k
    target = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target.add((i * m + j, j * p + k, k * n + i))

    def verify(U, V, W):
        """Strict verification: all target entries = 1, all non-target = 0."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # Check all target entries
        for (u, v, w) in target:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False
        # Check all non-target entries (full check for correctness)
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True

    # Try structured rank reduction by searching for valid decompositions
    # with rank slightly below schoolbook.
    # Strategy: start from schoolbook and try to "merge" pairs of terms or
    # find alternative sparse decompositions.
    
    import random
    random.seed(42)
    
    # Try a direct search: for each trial rank, attempt to find U,V,W
    # by starting from a perturbed schoolbook and doing local search.
    for trial_rank in range(rank_sb - 1, max(rank_sb - 20, 0), -1):
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            # Initialize with schoolbook-like structure but with some randomness
            U = [[0.0] * trial_rank for _ in range(n * m)]
            V = [[0.0] * trial_rank for _ in range(m * p)]
            W = [[0.0] * trial_rank for _ in range(p * n)]
            
            # Assign terms: each rank-1 term corresponds to a (i,j,k) triple
            # but we allow some terms to be "shared" or have different coefficients
            terms = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms.append((i, j, k))
            
            # Shuffle and try to pack terms into trial_rank slots
            random.shuffle(terms)
            # If trial_rank < len(terms), we need to merge some terms
            # This is complex; instead, try a simpler approach:
            # Use the first trial_rank terms as-is (schoolbook style)
            for r in range(min(trial_rank, len(terms))):
                i, j, k = terms[r]
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
            
            # Verify
            if verify(U, V, W):
                found = True
                break
        
        if found:
            best = {"U": U, "V": V, "W": W}
            best_rank = trial_rank
            break

    # Return best
    return best
# EVOLVE-BLOCK-END
