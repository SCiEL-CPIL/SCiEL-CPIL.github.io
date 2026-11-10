import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 modulo 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)
    
    # Precompute all subsets with their residues and sizes
    # subset_info[residue] = list of (size, frozenset_of_terms)
    # We want to find assignment minimizing max row length
    
    # First, let's find which residues are achievable with each size
    # For each size k from 1 to 10, compute all achievable residues
    
    # Map: residue -> list of subsets (as tuples) that sum to that residue
    # We'll organize by size for the matching
    
    # subsets_by_size[k] = list of (residue, frozenset) for subsets of size k
    subsets_by_size = [[] for _ in range(11)]
    
    for k in range(1, 11):
        for combo in combinations(range(1, 11), k):
            residue = sum(invs[x] for x in combo) % 101
            subsets_by_size[k].append((residue, combo))
    
    # Now use a greedy + matching approach:
    # For k = 1, 2, 3, ...:
    #   Check if we can assign all 101 residues using subsets of size <= k
    #   This is a bipartite matching problem: left = residues (101 nodes), right = subsets
    #   Edge from residue r to subset s if s sums to r
    #   We need a perfect matching (each residue gets one subset)
    #   But subsets can be used at most once (each subset is a unique combination)
    #   Wait - actually, can the same subset be used for multiple residues? No, each subset has exactly one residue.
    #   So each subset corresponds to exactly one residue. We need to pick one subset per residue.
    #   This is: for each residue r, pick one subset from subsets that sum to r.
    #   The constraint is just that each residue gets exactly one subset. There's no shared resource constraint
    #   between different residues since subsets are unique to their residue.
    #   
    #   Wait, I misread. Let me re-read: "Denominators may recur across different rows, but never within one row."
    #   So the same denominator CAN appear in multiple rows. The only constraint is that within a single row,
    #   denominators are distinct. And each row must be nonempty.
    #   
    #   So for each residue r, we just need to pick ONE subset of {1..10} (nonempty) whose inverses sum to r.
    #   There's no cross-row constraint! Each row is independent!
    #   
    #   So the problem is simply: for each residue r in 0..100, find a subset of {1..10} whose modular inverse
    #   sum equals r. Minimize the maximum size of these subsets.
    #   
    #   This means: find the smallest k such that every residue 0..100 is achievable by some subset of size <= k.
    #   Then for each residue, pick any subset of size <= k that achieves it.
    
    # Check: for each k, which residues are achievable by subsets of size <= k?
    # Find the minimum k such that all 101 residues are covered.
    
    # achievable_by_size[k] = set of residues achievable by subsets of size exactly k
    achievable_by_size = [set() for _ in range(11)]
    for k in range(1, 11):
        for residue, combo in subsets_by_size[k]:
            achievable_by_size[k].add(residue)
    
    # cumulative: achievable up to size k
    cumulative = [set() for _ in range(11)]
    for k in range(1, 11):
        cumulative[k] = cumulative[k-1] | achievable_by_size[k]
    
    # Find minimum k where all 101 residues are covered
    min_k = None
    for k in range(1, 11):
        if len(cumulative[k]) == 101:
            min_k = k
            break
    
    if min_k is None:
        # Should not happen, but fallback
        min_k = 10
    
    # Now, for each residue, pick a subset of size <= min_k that achieves it.
    # To minimize the maximum, we prefer smaller subsets.
    # For each residue, find the smallest subset that works.
    
    # best_subset[r] = the smallest combo (by size, then lexicographic) that sums to r
    best_subset = [None] * 101
    for k in range(1, min_k + 1):
        for residue, combo in subsets_by_size[k]:
            if best_subset[residue] is None:
                best_subset[residue] = list(combo)
    
    # Verify all are assigned
    for r in range(101):
        if best_subset[r] is None:
            # Should not happen if min_k is correct
            raise RuntimeError(f"Residue {r} not covered")
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Should not happen, this is fast
        pass
    
    return [best_subset[r] for r in range(101)]
