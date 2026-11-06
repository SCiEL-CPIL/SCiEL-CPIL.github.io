# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 80  # safety margin within 90s limit

    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]

    # Enumerate all non-empty subsets of {1..10}
    # For each subset: (size, residue, list_of_elements)
    all_subsets = []
    for mask in range(1, 1 << 10):
        elements = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(elements)
        residue = sum(inv[x - 1] for x in elements) % 101
        all_subsets.append((size, residue, elements))

    # Sort by size ascending (prefer smaller subsets)
    all_subsets.sort(key=lambda t: t[0])

    # Greedy assignment: for each subset (small to large), assign to the residue
    # that currently has the worst (largest) assigned subset, or is still uncovered.
    # We want to minimize the max row length.
    # 
    # Strategy: maintain for each residue the best (smallest) subset assigned so far.
    # Process subsets in order of increasing size. For each subset, if its residue
    # is not yet covered, assign it. If already covered with a larger subset, 
    # replace it (since we're going small to large, this won't happen).
    # But this just gives first-smallest per residue, which is the same as seed but sorted.
    #
    # Better structural approach: 
    # 1. First, for each residue, collect ALL subsets that map to it, sorted by size.
    # 2. The optimal assignment for minimizing max length is: for each residue, 
    #    pick the smallest subset. But that's trivial and doesn't conflict since
    #    each subset can only be used once... wait, the problem says denominators may
    #    recur across different rows. So each subset can be assigned to at most one row,
    #    but the same denominator can appear in multiple rows.
    #
    # Re-reading: "Denominators may recur across different rows, but never within one row."
    # This means: a subset (set of denominators) can be used in only one row. But the
    # same individual denominator can appear in multiple different rows (as part of 
    # different subsets).
    #
    # So the constraint is: each of the 1023 subsets can be assigned to at most one residue.
    # We need to assign one subset to each of the 101 residues, minimizing the max size.
    #
    # This is an assignment problem! We have 101 residues and 1023 subsets. Each subset
    # maps to exactly one residue. We need to pick 101 subsets (one per residue) such that
    # each residue gets exactly one subset, and we minimize the max size.
    #
    # Greedy: sort all subsets by size. For each size k from 1 to 10, try to assign
    # subsets of size k to residues that don't yet have an assignment. Use bipartite
    # matching (or greedy) to maximize coverage.
    #
    # Let's do: for k = 1, 2, ..., 10:
    #   Consider all unassigned subsets of size k.
    #   For each such subset, it covers residue r.
    #   Assign it to residue r if r is still uncovered.
    #   This greedy may not be optimal but is fast.
    #
    # Even better: for each k, build a bipartite graph between size-k subsets and 
    # uncovered residues, and find a maximum matching. Assign matched pairs.
    
    # Group subsets by size
    subsets_by_size = defaultdict(list)
    for size, residue, elements in all_subsets:
        subsets_by_size[size].append((residue, elements))
    
    # For each size, group by residue
    # subsets_by_size[k][r] = list of element-lists for subsets of size k mapping to residue r
    
    assignment = {}  # residue -> elements list
    used_subsets = set()  # (residue, tuple(elements)) to track used subsets
    
    for k in range(1, 11):
        if time.time() > deadline:
            break
        # Get all subsets of size k
        size_k_subsets = subsets_by_size[k]
        if not size_k_subsets:
            continue
        
        # Build adjacency: for each uncovered residue, which size-k subsets can cover it
        uncovered = [r for r in range(101) if r not in assignment]
        if not uncovered:
            break
        
        # For greedy: just iterate and assign
        # For each residue in uncovered, pick the first available size-k subset
        # To be more systematic, let's do a simple greedy matching
        
        # Map residue -> list of (elements) for size k
        residue_to_elements = defaultdict(list)
        for residue, elements in size_k_subsets:
            residue_to_elements[residue].append(elements)
        
        # Greedy: for each uncovered residue (in some order), assign the first available
        # Sort uncovered by number of options (fewest first) for better greedy
        uncovered_sorted = sorted(uncovered, key=lambda r: len(residue_to_elements.get(r, [])))
        
        for r in uncovered_sorted:
            if r in assignment:
                continue
            if time.time() > deadline:
                break
            options = residue_to_elements.get(r, [])
            if options:
                # Pick the first one (they're all size k, so any works)
                assignment[r] = options[0]
                # Mark as used (though we don't strictly need to since each subset 
                # maps to exactly one residue, and we're picking one per residue)
                # Actually, multiple subsets can map to the same residue, but we only
                # need one per residue. And each subset is uniquely identified by its
                # element set. Since we're picking one subset per residue, and subsets
                # are distinct, there's no conflict.
    
    # Check if all residues are covered
    if len(assignment) < 101:
        # Fallback: fill remaining with any available subset
        for r in range(101):
            if r not in assignment:
                # Find any subset that maps to r
                for size, residue, elements in all_subsets:
                    if residue == r:
                        assignment[r] = elements
                        break
    
    # Now try to improve: local search to reduce max length
    # Current max length
    def get_max_len(assignment):
        return max(len(v) for v in assignment.values())
    
    current_max = get_max_len(assignment)
    
    # Try to reduce: for each residue with the largest subset, try to find a smaller
    # subset for that residue that isn't already used by another residue.
    # Since subsets are uniquely mapped to residues, we just need to find a smaller
    # subset for that residue.
    
    # Precompute: for each residue, all subsets sorted by size
    residue_subsets = defaultdict(list)
    for size, residue, elements in all_subsets:
        residue_subsets[residue].append((size, elements))
    for r in residue_subsets:
        residue_subsets[r].sort(key=lambda t: t[0])
    
    # Track which subsets are currently used (by residue)
    used_elements = set()
    for r, elems in assignment.items():
        used_elements.add(tuple(elems))
    
    # Try to improve each residue's assignment to a smaller subset
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Process residues in order of current size (largest first)
        residues_sorted = sorted(range(101), key=lambda r: len(assignment[r]), reverse=True)
        for r in residues_sorted:
            if time.time() > deadline:
                break
            current_size = len(assignment[r])
            # Look for a smaller subset for residue r
            for size, elements in residue_subsets[r]:
                if size >= current_size:
                    break
                elem_tuple = tuple(elements)
                if elem_tuple not in used_elements:
                    # Replace
                    used_elements.discard(tuple(assignment[r]))
                    assignment[r] = elements
                    used_elements.add(elem_tuple)
                    improved = True
                    break
    
    return [assignment[r] for r in range(101)]
# EVOLVE-BLOCK-END
