import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute inverses of 1..10 modulo 101
    invs = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all subsets (non-empty) with their sums
    # We want to minimize the maximum row length.
    # Strategy: For each possible max length K (1 to 10), check if we can cover all residues.
    # If yes, assign to each residue the smallest possible subset (length <= K).
    # To be optimal, we should find the minimal K first.
    
    # Generate all subsets and their sums
    # subset -> (sum, denominators)
    all_subsets = {}
    
    # We can iterate by size to prioritize smaller subsets
    subsets_by_size = {k: [] for k in range(1, 11)}
    
    for k in range(1, 11):
        for comb in combinations(range(1, 11), k):
            s = sum(invs[x-1] for x in comb) % 101
            subsets_by_size[k].append((s, comb))
            
    # Find minimal K such that all residues are covered by subsets of size <= K
    # And then assign the smallest subset for each residue.
    
    # We'll try K=1, 2, ...
    # For a given K, we can cover a residue r if there exists a subset of size <= K with sum r.
    # If multiple, we pick the one with smallest size.
    
    # Let's precompute for each residue the smallest subset that maps to it.
    # This is independent of K if we just want the global minimum size per residue.
    # But the objective is "Minimize the largest row length".
    # So if we pick the smallest subset for each residue independently, the max length is the max of those sizes.
    # Is it possible that picking a slightly larger subset for one residue allows us to use a smaller max length overall?
    # No, because the constraint is just "row r contains distinct denominators... whose sum is r".
    # There is no coupling between rows other than the objective function max(len(row)).
    # So the optimal strategy is: for each residue, find the subset with the minimum size that sums to that residue.
    # Then the answer is max(len(subset_r) for all r).
    
    # Wait, is that true?
    # "Minimize the largest row length."
    # If I choose subset S_r for residue r, the cost is max(|S_r|).
    # Since the choices for different r are independent (denominators can recur across rows),
    # the optimal choice for each r is indeed the subset with the minimum cardinality that sums to r.
    
    # So the algorithm is:
    # 1. For each residue r in 0..100, find the subset of {1..10} with minimum size that sums to r mod 101.
    # 2. Return these subsets.
    # 3. The score is the maximum size among these subsets.
    
    # Let's verify if "minimum size subset for each residue" is always possible.
    # Since we have 2^10 - 1 = 1023 subsets and 101 residues, by Pigeonhole, many residues have multiple representations.
    # We just need at least one.
    
    best_rep = {}
    
    # Iterate by size to ensure minimum cardinality for each residue
    for k in range(1, 11):
        # Check deadline
        if time.time() > deadline:
            break
            
        for s, comb in subsets_by_size[k]:
            if s not in best_rep:
                best_rep[s] = comb
                if len(best_rep) == 101:
                    break
        if len(best_rep) == 101:
            break
            
    # Check if all residues are covered
    if len(best_rep) < 101:
        # This should not happen, but handle gracefully
        # Find any representation for missing residues
        for r in range(101):
            if r not in best_rep:
                # Search all subsets
                for k in range(1, 11):
                    for s, comb in subsets_by_size[k]:
                        if s == r:
                            best_rep[r] = comb
                            break
                    if r in best_rep:
                        break
                if r not in best_rep:
                    # Should never happen
                    best_rep[r] = [1]
                    
    return [best_rep[r] for r in range(101)]
