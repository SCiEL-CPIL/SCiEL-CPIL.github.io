import time

def construct():
    # Precompute inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # Precompute all subsets
    # subset_info: list of (size, residue, terms_tuple)
    subset_info = []
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        size = 0
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                terms.append(i)
                res += inv[i]
                size += 1
        res %= 101
        subset_info.append((size, res, tuple(terms)))
        
    # Sort by size to easily check coverage for increasing L
    subset_info.sort(key=lambda x: x[0])
    
    # We need to find the minimal L such that all residues are covered by subsets of size <= L
    # And then for each residue, pick the subset with minimal size (<= L)
    
    # We can just iterate L from 1 to 10
    for L in range(1, 11):
        # Map residue to the smallest size subset (terms) that covers it, with size <= L
        best_subset_for_residue = {}
        
        for size, res, terms in subset_info:
            if size > L:
                break
            # If we already have a subset for this residue, we keep the one with smaller size
            # Since we are iterating in increasing size, the first one we see for a residue is the smallest.
            # However, multiple subsets of the same size might exist. Any is fine.
            if res not in best_subset_for_residue:
                best_subset_for_residue[res] = terms
                
        # Check if all 101 residues are covered
        if len(best_subset_for_residue) == 101:
            # Construct the result
            result = []
            for r in range(101):
                result.append(list(best_subset_for_residue[r]))
            return result
            
    # Fallback (should not happen)
    return [list(subset_info[0][2])] * 101
