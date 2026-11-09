"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precompute modular inverses for 1..10
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)
        
    # Map residue -> list of (length, subset)
    # We want to find the minimum max length
    residue_to_subsets = {r: [] for r in range(101)}
    
    # Iterate through all non-empty subsets of {1..10}
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + invs[x]) % 101
        
        # Store the subset and its length
        residue_to_subsets[s].append((len(terms), terms))
        
    # Find the minimal k such that every residue has a subset of length <= k
    # We can just sort subsets by length for each residue and pick the shortest
    # Actually, we just need to know if a solution exists for a given max_length.
    # Let's find the global minimum max_length.
    
    # For each residue, find the minimum length subset available
    min_lengths = []
    for r in range(101):
        if not residue_to_subsets[r]:
            # Should not happen as we have 1023 subsets for 101 residues
            # But just in case, if a residue is missing, we can't cover it.
            # However, the problem states "Every residue must be represented", 
            # implying it's possible. With 1023 subsets, it's very likely.
            min_lengths.append(100) # Penalty
        else:
            # Find the subset with the minimum length
            min_len = min(len_s for len_s, _ in residue_to_subsets[r])
            min_lengths.append(min_len)
            
    # The optimal maximum length is the max of these minimums
    optimal_max_len = max(min_lengths)
    
    # Now construct the result using subsets of length <= optimal_max_len
    # We can just pick the shortest subset for each residue
    result = []
    for r in range(101):
        # Pick the subset with the minimum length
        # If there are multiple, any is fine
        best_subset = None
        best_len = float('inf')
        for len_s, terms in residue_to_subsets[r]:
            if len_s < best_len:
                best_len = len_s
                best_subset = terms
        # If for some reason no subset was found (shouldn't happen), fallback to any
        if best_subset is None:
            best_subset = residue_to_subsets[r][0][1]
        result.append(best_subset)
        
    return result
# EVOLVE-BLOCK-END
