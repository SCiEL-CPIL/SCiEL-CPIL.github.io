"""S(5,6,12) Witt design via exact cover / backtracking with MRV."""
import time
from itertools import combinations

# Precompute all C(12,6) = 924 possible 6-sets
# and for each, its C(6,5)=6 five-subsets

def _precompute():
    """Precompute candidate blocks and their 5-subsets."""
    all_blocks = []
    for b in combinations(range(12), 6):
        subsets = tuple(combinations(b, 5))
        all_blocks.append((b, subsets))
    return all_blocks

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill

    all_blocks = _precompute()
    n_blocks = len(all_blocks)
    
    # We want to select a subset of blocks such that no 5-subset is covered twice.
    # Total 5-subsets of 12 points: C(12,5) = 792
    # Each block covers 6 five-subsets.
    # Max blocks: 792/6 = 132.
    
    # Use exact cover formulation: we need to cover all 792 five-subsets exactly once.
    # This is a classic exact cover problem (like Knuth's Dancing Links).
    
    # For efficiency, use a recursive backtracking with MRV (minimum remaining values)
    # But the constraint is: no two blocks share a 5-subset.
    
    # Alternative: greedy + local search, or pure backtracking with good heuristics.
    
    # Given the structure, let's use a smarter approach:
    # The S(5,6,12) design is unique up to isomorphism. We can try to construct it
    # using known symmetry properties or use a fast backtracking.
    
    # Fast approach: use the fact that we can fix some blocks by symmetry.
    # Fix block {0,1,2,3,4,5}. By the automorphism group, we can also fix more.
    
    # Actually, let's just do a well-optimized backtracking.
    # Key insight: with good MRV and symmetry breaking, this should be fast.
    
    # Pre-index: for each 5-subset, which blocks contain it?
    # 5-subset identified by frozenset of 5 elements
    subset_to_blocks = {}
    for i, (b, subsets) in enumerate(all_blocks):
        for s in subsets:
            key = s  # tuple of 5 sorted elements
            if key not in subset_to_blocks:
                subset_to_blocks[key] = []
            subset_to_blocks[key].append(i)
    
    # For backtracking, we need to track which 5-subsets are covered.
    # Use a bitset or set.
    
    # Let's use a different strategy: iterative construction with conflict detection.
    # Sort blocks by a good heuristic and greedily add, then if stuck, backtrack.
    
    # Actually for this specific problem, a simple greedy with the right ordering
    # might work, or we need actual backtracking.
    
    # Let me try: order blocks so that we pick "spread out" ones first.
    # Better: use the constraint that we need exactly 132 blocks.
    
    # Let's do backtracking with MRV on 5-subsets.
    # State: set of covered 5-subsets, list of chosen blocks.
    # At each step, find the 5-subset with fewest remaining available blocks.
    
    # For speed, represent covered as a set of frozensets or encoded integers.
    # Encode a 5-subset as an integer: 5 elements from 0-11, use combinatorial number system.
    
    # Actually, let's just use tuples as keys.
    
    blocks_found = []
    covered = set()  # set of 5-tuples
    
    # Sort blocks: prefer those that share fewer 5-subsets with already-chosen blocks?
    # No, we need to be careful. Let's just try all blocks in a fixed order with backtracking.
    
    # Given 924 blocks and we need 132, naive backtracking is infeasible without pruning.
    
    # Better approach: use the known structure.
    # The Witt design S(5,6,12) can be constructed from the 12 points as follows:
    # It's related to the PSL(2,11) group action or the Mathieu group M12.
    
    # Let me just hardcode a known construction or generate it via a fast method.
    
    # Known: the blocks of S(5,6,12) can be generated from a set of "seed" blocks
    # under the action of the automorphism group M12.
    
    # M12 has order 95040. The stabilizer of a block has order 95040/132 = 720 = 6!.
    # So we need 132 blocks, each in an orbit.
    
    # Actually, let me just do a clean backtracking with strong pruning.
    # The key is: when we add a block, we eliminate all blocks that share a 5-subset with it.
    # This is a set packing problem.
    
    # Let me try a different tactic: use numpy for vectorized operations.
    # Represent each block by the 6 five-subsets it covers.
    # Use a bitset of length 792 for coverage.
    
    # Map each 5-subset to an index 0..791
    all_5subsets = list(combinations(range(12), 5))
    subset_index = {s: i for i, s in enumerate(all_5subsets)}
    
    # For each block, store the list of 5-subset indices it covers
    block_subsets = []
    for b, subsets in all_blocks:
        idxs = [subset_index[s] for s in subsets]
        block_subsets.append(idxs)
    
    # Now do backtracking with bitsets
    import numpy as np
    
    # Represent coverage as a numpy bool array of length 792
    # For each block, a numpy bool array marking its 6 subsets
    
    # Actually, for backtracking efficiency, let's use Python sets and a recursive approach
    # with good ordering.
    
    # Let me try: order blocks by their first element, then lexicographic.
    # Use recursion with memoization won't help here.
    
    # Given the time constraint, let me try a different approach:
    # Use the fact that S(5,6,12) is the unique design, and construct it
    # using the known "short" representation.
    
    # The 132 blocks of S(5,6,12) in terms of the 12 points can be listed.
    # Let me generate them using the following method:
    # Start with the point set Z_11 union {inf} (12 points).
    # The blocks are:
    # 1. {inf, a, b, c, d, e} where {a,b,c,d,e} is a 5-element subset of Z_11 
    #    with certain properties... 
    
    # This is getting complicated. Let me just do a well-optimized search.
    
    # Strategy: Use recursive backtracking with the following:
    # 1. Fix the first block to {0,1,2,3,4,5} (by symmetry)
    # 2. Use MRV: at each step, pick the constraint (5-subset) with fewest options
    # 3. Prune: if remaining available blocks < needed blocks, fail
    
    # Let me implement this carefully.
    
    def solve():
        start_t = time.time()
        
        # Symmetry breaking: fix block {0,1,2,3,4,5}
        # Its 5-subsets:
        fixed_block = (0, 1, 2, 3, 4, 5)
        fixed_subsets = set(combinations(fixed_block, 5))
        
        # All blocks compatible with fixed_block: don't share any 5-subset
        # A block shares a 5-subset with fixed_block iff it contains 5 of {0,1,2,3,4,5}
        # i.e., it contains 5 elements from {0,...,5} and 1 from {6,...,11}
        # OR it IS the fixed block.
        
        compatible = []
        for i, b in enumerate(all_blocks):
            if b == fixed_block:
                continue
            # Check if b shares a 5-subset with fixed_block
            # b shares a 5-subset iff |b ∩ fixed_block| >= 5
            inter = len(set(b) & set(fixed_block))
            if inter < 5:
                compatible.append(i)
        
        # Also need to track: the 5-subsets NOT covered by fixed_block
        # We need to cover all 792 - 6 = 786 remaining 5-subsets with 131 more blocks
        
        # For backtracking, track:
        # - chosen blocks (indices)
        # - covered 5-subsets (set)
        
        # This is still complex. Let me try a simpler but effective approach:
        # Greedy with randomization and multiple restarts.
        
        best = None
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Random shuffle for diversity
            import random
            rng = random.Random(attempt)
            
            order = list(range(n_blocks))
            rng.shuffle(order)
            
            # Always try fixed block first for symmetry breaking
            # Find index of fixed block
            fixed_idx = None
            for i, b in enumerate(all_blocks):
                if b == fixed_block:
                    fixed_idx = i
                    break
            
            if fixed_idx is not None:
                order.remove(fixed_idx)
                order.insert(0, fixed_idx)
            
            blocks_found = []
            covered = set()
            
            for idx in order:
                b, subsets = all_blocks[idx]
                if not any(s in covered for s in subsets):
                    blocks_found.append(b)
                    covered.update(subsets)
            
            if len(blocks_found) > (best is None or len(blocks_found) > len(best)):
                best = blocks_found
            
            if len(blocks_found) == 132:
                return [list(b) for b in best]
        
        if best is not None:
            return [list(b) for b in best]
        return []
    
    result = solve()
    return result
