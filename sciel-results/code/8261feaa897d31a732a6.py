import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = 80.0

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Enumerate all non-empty subsets, group by size
    # subsets_by_size[size] = list of (residue, terms)
    subsets_by_size = [[] for _ in range(11)]
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        subsets_by_size[len(terms)].append((s, terms))

    # Find minimum L such that all 101 residues can be covered by subsets of size <= L
    # For each L, we need to check if there exists a perfect matching
    # between residues and subsets (where each residue is matched to a subset
    # that sums to it modulo P, and each subset is used at most once)
    
    # Since subsets can be reused across different residues (denominators may recur
    # across different rows), we actually just need: for each residue r, there exists
    # at least one subset of size <= L that sums to r mod P.
    # This is NOT a matching problem - it's simply a coverage problem.
    
    # So we just need to find the smallest L such that the union of residues
    # achievable by subsets of size 1..L covers all 101 residues.
    
    assignment = {}
    for L in range(1, 11):
        for residue, terms in subsets_by_size[L]:
            if residue not in assignment:
                assignment[residue] = terms
        
        if len(assignment) == P:
            # All residues covered. Now refine: for each residue, pick the 
            # subset with minimum size (and break ties by minimum sum of denominators)
            # to minimize the maximum row length further if possible.
            
            # Actually, since we're using the smallest L that covers all, 
            # we should pick for each residue the best subset among sizes 1..L
            # that achieves that residue.
            
            best_for_residue = {}
            for size in range(1, L + 1):
                for residue, terms in subsets_by_size[size]:
                    if residue not in best_for_residue:
                        best_for_residue[residue] = terms
                    else:
                        # Prefer smaller size, then smaller sum of denominators
                        existing = best_for_residue[residue]
                        if len(terms) < len(existing):
                            best_for_residue[residue] = terms
                        elif len(terms) == len(existing) and sum(terms) < sum(existing):
                            best_for_residue[residue] = terms
            
            result = [best_for_residue[r] for r in range(P)]
            return result

    # Fallback: should never reach here since L=10 includes all subsets
    # which must cover all residues (there are 1023 subsets for 101 residues)
    result = [assignment[r] if r in assignment else [1] for r in range(P)]
    return result
