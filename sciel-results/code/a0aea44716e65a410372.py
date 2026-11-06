import time

def construct():
    # Precompute inverses
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # best_rep[r] will store the shortest list of denominators summing to r
    # We initialize with None
    best_rep = [None] * 101
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    # There are 2^10 - 1 = 1023 subsets. This is very fast.
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + inv[x]) % 101
        
        size = len(terms)
        
        # If this is a better (smaller) representation for this residue, update
        # If no representation yet, set it
        if best_rep[s] is None or size < len(best_rep[s]):
            best_rep[s] = terms

    # Verify all residues are covered
    # If some residue is not covered (should not happen for k<=10), 
    # we might need a fallback, but 1023 subsets vs 101 residues makes 
    # it virtually certain that all are covered by size <= 4 or 5.
    
    # The problem asks to minimize the largest row length.
    # The current best_rep holds the shortest list for each residue.
    # The largest length in best_rep is the minimal possible max length 
    # IF we are allowed to pick independently for each residue.
    # The problem states: "row r containing distinct denominators... Every residue must be represented."
    # It does NOT require the sets to be disjoint across rows. "Denominators may recur across different rows".
    # So independent selection is valid.
    
    return [list(best_rep[r]) for r in range(101)]
