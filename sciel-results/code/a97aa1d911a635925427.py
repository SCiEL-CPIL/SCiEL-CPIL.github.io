"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(inv[x - 1] for x in terms) % 101
        all_subsets.append((size, residue, mask))
    
    # Group subsets by residue, sorted by size
    subsets_by_residue = defaultdict(list)
    for size, residue, mask in all_subsets:
        subsets_by_residue[residue].append((size, mask))
    
    # Sort each list by size
    for r in subsets_by_residue:
        subsets_by_residue[r].sort(key=lambda x: x[0])
    
    # We want to assign to each residue 0..100 a subset such that:
    # - No two subsets share a denominator (i.e., masks are disjoint)
    # - Minimize the maximum size
    
    # This is a constraint satisfaction / matching problem.
    # Since 101 residues and 10 denominators, we can use backtracking with pruning.
    # Key insight: each denominator can be used in at most one subset.
    # So we need 101 subsets that are pairwise disjoint in terms of denominators.
    # But wait - we only have 10 denominators, and each subset uses at least 1.
    # So at most 10 subsets can be non-empty if they must be pairwise disjoint!
    # That's a contradiction with needing 101 residues.
    
    # Let me re-read the problem: "Denominators may recur across different rows, but never within one row."
    # So denominators CAN be reused across different rows. The constraint is only within a row.
    # So each row just needs distinct denominators, but the same denominator can appear in multiple rows.
    
    # This changes everything. We just need for each residue r, find a subset of {1..10} 
    # whose inverse sum is r mod 101. We want to minimize the maximum subset size.
    
    # Since denominators can recur, each residue can independently pick its best subset.
    # So the answer is: for each residue, find the smallest subset that sums to it.
    # The max row length is the maximum over all residues of the minimum subset size.
    
    # Let's find for each residue the minimum size subset that achieves it.
    best_size_for_residue = [11] * 101  # 11 = impossible
    best_mask_for_residue = [0] * 101
    
    # Iterate by size to find minimum
    for size in range(1, 11):
        for mask in range(1, 1 << 10):
            if bin(mask).count('1') != size:
                continue
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            residue = sum(inv[x - 1] for x in terms) % 101
            if best_size_for_residue[residue] > size:
                best_size_for_residue[residue] = size
                best_mask_for_residue[residue] = mask
    
    # Check if all residues are covered
    max_row_len = max(best_size_for_residue)
    
    # Build the result
    result = []
    for r in range(101):
        mask = best_mask_for_residue[r]
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        result.append(terms)
    
    return result
# EVOLVE-BLOCK-END
