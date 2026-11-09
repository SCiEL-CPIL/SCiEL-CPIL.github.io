"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(11)]
    
    # For each residue, find the minimum-size subset that achieves it
    # We iterate by subset size to find the optimal max row length
    # There are C(10,k) subsets of size k
    
    # First pass: compute for each residue the minimum subset size
    min_size = [None] * 101  # min_size[r] = smallest k such that some k-subset sums to r
    min_subset = [None] * 101  # the actual subset achieving min_size[r]
    
    # Enumerate all nonempty subsets, grouped by size
    from itertools import combinations
    
    # Precompute all subsets by size
    all_subsets_by_size = {}
    for k in range(1, 11):
        subs = []
        for combo in combinations(range(1, 11), k):
            subs.append(combo)
        all_subsets_by_size[k] = subs
    
    # For each size k from 1 to 10, assign to residues not yet covered
    for k in range(1, 11):
        if time.time() > deadline:
            break
        for combo in all_subsets_by_size[k]:
            s = sum(inv[x] for x in combo) % 101
            if min_size[s] is None:
                min_size[s] = k
                min_subset[s] = list(combo)
        # Check if all residues are covered
        if all(ms is not None for ms in min_size):
            break
    
    # Build the result: row r gets min_subset[r]
    # If any residue is still None (shouldn't happen with 1023 subsets),
    # fall back to any representation
    result = []
    for r in range(101):
        if min_subset[r] is not None:
            result.append(min_subset[r])
        else:
            # Fallback: brute force find any subset
            found = None
            for k in range(1, 11):
                for combo in all_subsets_by_size[k]:
                    s = sum(inv[x] for x in combo) % 101
                    if s == r:
                        found = list(combo)
                        break
                if found:
                    break
            if found is None:
                # Should never happen, but provide a valid nonempty list
                found = [1]
            result.append(found)
    
    return result

# EVOLVE-BLOCK-END
