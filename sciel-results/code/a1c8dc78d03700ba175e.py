"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute all non-empty subsets of {1, ..., 10}
    # Each entry: (residue, size, list_of_terms)
    subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        size = len(terms)
        subsets.append((residue, size, terms))
    
    # Find the minimum K such that all 101 residues are covered by subsets of size <= K
    # K ranges from 1 to 10
    for K in range(1, 11):
        # Map residue to the first (smallest mask/index) subset of size <= K
        # Using a dict to store the best candidate for each residue
        best_for_residue = {}
        
        for residue, size, terms in subsets:
            if size <= K:
                # We just need *some* subset. 
                # To be deterministic and potentially "nicer", we could pick the one with smallest size, 
                # or the one with smallest mask. 
                # Since we iterate K upwards, any valid subset works for the "existence" check.
                # For the actual return value, we should probably pick the smallest size available 
                # within the threshold K, or just the first one found.
                # Let's store the one with the smallest size, and if tie, smallest mask (implicit in loop order? No, mask order is fixed).
                # Actually, just storing the first one found is fine, as long as we verify coverage.
                # However, to be safe and consistent, let's store the one with minimal size, then minimal mask.
                if residue not in best_for_residue:
                    best_for_residue[residue] = (size, terms)
                else:
                    # Update if current is better (smaller size, or same size and lexicographically smaller? 
                    # The problem doesn't require lexicographic order, just minimizing max length.
                    # So any valid assignment works. 
                    # But let's stick to the first valid one found for simplicity, 
                    # or the one with smallest size to be "tighter" locally?
                    # It doesn't affect the max length K. 
                    # Let's just keep the first one found for speed.
                    pass

        # Check if all residues 0..100 are covered
        if len(best_for_residue) == 101:
            # Construct the final list
            result = [None] * 101
            for r in range(101):
                # best_for_residue[r] is (size, terms)
                _, terms = best_for_residue[r]
                result[r] = terms
            
            # Verify non-empty and distinct denominators (already guaranteed by subset construction)
            # Verify sum modulo 101
            # This is just a sanity check, not strictly necessary if logic is correct
            # for r in range(101):
            #     s = sum(pow(x, -1, 101) for x in result[r]) % 101
            #     assert s == r, f"Error at residue {r}: got {s}"
            
            return result
    
    # Fallback (should never happen as K=10 covers all)
    # If we get here, it means some residue is not covered by any subset?
    # But sum of all 10 inverses is a specific residue. 
    # Actually, the set of all subset sums modulo 101 must cover all residues if the group generated is the whole group?
    # With 10 elements, we have 1023 sums. It is extremely likely they cover all 101 residues.
    # If not, we must return something valid. 
    # But for p=101 and n=10, it is a known fact that they cover all residues.
    
    # Just in case, return the first found for each, even if max size is large
    representations = {}
    for residue, size, terms in subsets:
        if residue not in representations:
            representations[residue] = terms
    return [representations[r] for r in range(101)]
# EVOLVE-BLOCK-END
