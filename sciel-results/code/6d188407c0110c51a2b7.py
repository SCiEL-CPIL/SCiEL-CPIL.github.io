"""Stronger covering design algorithm for C(v, k, t) with deadline guard."""
import itertools
import time
import random

def construct(v, k, t):
    # Deadline guard: must return well before 150s wall-clock
    deadline = time.time() + 140  # 10s safety margin
    
    # Total number of t-subsets
    total_t_subsets = 1
    for i in range(t):
        total_t_subsets *= (v - i)
    for i in range(1, t + 1):
        total_t_subsets //= i
    
    # We'll track uncovered t-subsets as a set of tuples
    # For C(19,8,5): C(19,5) = 11628 subsets
    
    all_t_subsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Precompute: for each point, the list of (t-1)-subsets it can extend
    # Actually, let's use a smarter greedy with better gain computation
    
    # For the initial greedy: pick the first uncovered t-subset, then build a block
    # around it by adding points that cover the most uncovered t-subsets
    
    # Optimization: maintain a count of how many uncovered t-subsets each point-set
    # would cover. But that's expensive. Instead, use a two-phase approach:
    # Phase 1: Greedy construction with good heuristics
    # Phase 2: Local search / removal pass to reduce block count
    
    # Better approach: Use a priority-based greedy where at each step we try
    # multiple candidate blocks and pick the one that covers the most uncovered subsets.
    
    # For efficiency, let's use the following strategy:
    # 1. Start with the lexicographically smallest uncovered t-subset as seed
    # 2. Greedily add points to maximize coverage of uncovered t-subsets
    # 3. After building the block, remove any covered t-subsets from uncovered
    # 4. Repeat until all covered
    # 5. Then do a removal pass: try removing each block and see if coverage is maintained
    
    while uncovered and time.time() < deadline - 10:
        # Pick the uncovered t-subset with the fewest covering options (hardest to cover)
        # For simplicity, pick the first one (lexicographic)
        seed = min(uncovered)
        
        # Build a block containing seed
        block = set(seed)
        remaining_slots = k - t
        
        # Greedily add points
        for _ in range(remaining_slots):
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset S is covered by block ∪ {x} if S ⊆ (block ∪ {x})
                # and x ∈ S (otherwise it was already covered by block)
                # More precisely: S is newly covered if S ⊆ (block ∪ {x}) and x ∈ S
                # i.e., S \ {x} ⊆ block and x ∈ S
                gain = 0
                # For each (t-1)-subset of block, check if adding x forms a covered t-subset
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is not None and best_gain > 0:
                block.add(best_point)
            elif best_point is not None:
                # No gain, just add a random point
                available = [x for x in range(v) if x not in block]
                if available:
                    block.add(random.choice(available))
                else:
                    break
            else:
                break
        
        # Convert to sorted list
        block_list = sorted(block)
        
        # Only add if it actually covers some uncovered t-subsets
        new_covered = set(itertools.combinations(block_list, t)) & uncovered
        if new_covered:
            blocks.append(block_list)
            uncovered -= new_covered
        else:
            # This block doesn't help, skip it (shouldn't happen often)
            pass
    
    # Removal pass: try to remove blocks that are redundant
    if time.time() < deadline - 5 and len(blocks) > 1:
        # Build a coverage map: for each t-subset, which blocks cover it
        # For efficiency, only do this if we have time
        coverage = {}
        for i, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                if ts not in coverage:
                    coverage[ts] = []
                coverage[ts].append(i)
        
        # Try removing blocks in reverse order (later blocks may be less critical)
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            # Check if any t-subset is only covered by block i
            can_remove = True
            for ts in itertools.combinations(blocks[i], t):
                if len(coverage.get(ts, [])) <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove block i
                for ts in itertools.combinations(blocks[i], t):
                    coverage[ts].remove(i)
                blocks.pop(i)
    
    return blocks
