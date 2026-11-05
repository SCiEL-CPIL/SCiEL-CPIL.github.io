import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    P = 101
    # Precompute inverses
    inv = {x: pow(x, -1, P) for x in range(1, 11)}
    
    # Precompute all non-empty subsets with their residue and size
    all_subsets = []  # list of (residue, size, subset_list)
    for mask in range(1, 1 << 10):
        subset = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
        r = s % P
        sz = len(subset)
        all_subsets.append((r, sz, subset))
    
    # Group subsets by residue
    subsets_by_residue = [[] for _ in range(P)]
    for r, sz, subset in all_subsets:
        subsets_by_residue[r].append((sz, subset))
    
    # For each residue, sort subsets by size
    for r in range(P):
        subsets_by_residue[r].sort(key=lambda x: x[0])
    
    # Minimax feasibility: find smallest K such that all residues can be
    # covered using subsets of size <= K
    # Since denominators can recur across rows, each residue independently
    # needs at least one subset of size <= K.
    # So the minimum possible max row length is:
    #   max over all residues r of (min size of subset achieving r)
    # This is exactly what the original approach computes!
    
    # But wait - let me reconsider. The problem says "minimize the largest row length."
    # Each row r must contain distinct denominators in 1..10 whose inverses sum to r mod 101.
    # Denominators may recur across different rows.
    # So each row is independently chosen. The max row length is max over r of len(row_r).
    # To minimize this, for each r we want the smallest possible subset.
    # So the optimal max row length = max_r (min subset size for residue r).
    
    # The original approach already finds this! Let me verify and just return it.
    best = [None] * P
    for r in range(P):
        if subsets_by_residue[r]:
            best[r] = subsets_by_residue[r][0]  # smallest size subset
        else:
            best[r] = (10, list(range(1, 11)))  # fallback
    
    # Verify all residues are covered
    for r in range(P):
        if best[r] is None:
            # Should not happen, but fallback
            for r2, sz, subset in all_subsets:
                if r2 == r:
                    best[r] = (sz, subset)
                    break
            if best[r] is None:
                best[r] = (1, [1])
    
    return [best[r][1] for r in range(P)]
# EVOLVE-BLOCK-END
