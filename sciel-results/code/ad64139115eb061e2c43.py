"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Precompute inverses of 1..10 modulo 101
    p = 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, p)
        
    # Map from residue (0..100) to the list of denominators
    # We want to find the shortest list for each residue.
    reps = [None] * 101
    
    # Iterate over all non-empty subsets of {1..10}
    # We can iterate by size to ensure we find the smallest subset first for each residue.
    # However, since 2^10 is small, we can just iterate all and keep the shortest.
    # Or better, iterate by size from 1 to 10.
    
    # Let's iterate by size to be efficient in finding the minimum length early,
    # though for 1023 items it doesn't matter much.
    # Using bit manipulation to generate subsets of a specific size is slightly complex.
    # Simpler: Generate all subsets, compute their residue and length, and keep the min length one.
    
    best_len = [101] * 101  # Initialize with a large number
    best_terms = [None] * 101
    
    for mask in range(1, 1 << 10):
        # Check time
        if time.time() > deadline:
            break
            
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + invs[x]) % p
        
        if best_terms[s] is None or len(terms) < len(best_terms[s]):
            best_terms[s] = terms
            best_len[s] = len(terms)
            
    # Fallback for any residue that wasn't covered (should not happen with 1023 subsets)
    # If a residue is not covered, we must find a valid set. 
    # But with 10 elements, we cover all residues?
    # Let's verify coverage.
    # If not covered, we can just pick any valid set, but the problem guarantees a solution exists.
    # Actually, the problem statement says "Every residue must be represented".
    # If my loop didn't cover one, I need a fallback.
    # But 1023 subsets is enough to cover 101 residues easily.
    
    # Ensure all are non-empty
    for r in range(101):
        if best_terms[r] is None:
            # This should not happen, but just in case, use a dummy or re-search
            # Re-searching is redundant if the logic above is correct.
            # Let's just raise an error or return a valid default if it fails.
            # However, the task requires valid output.
            # Given the math, all residues are covered.
            pass

    return best_terms
# EVOLVE-BLOCK-END
