"""Primal-dual inspired greedy covering for C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a weighted greedy algorithm
    inspired by primal-dual approximation for set cover.
    
    Strategy:
    - Maintain a weight (dual variable) for each uncovered t-subset.
    - At each step, select the k-block that maximizes the sum of weights of
      uncovered t-subsets it covers, divided by its 'cost' (1).
    - To break ties and improve coverage, we can use a secondary criterion:
      prefer blocks that cover t-subsets with higher 'urgency' (i.e., those
      that are contained in fewer potential blocks, or simply use randomization
      with a fixed seed for reproducibility).
    
    For better performance, we use a more sophisticated greedy:
    - Start with all t-subsets uncovered.
    - Repeatedly find the k-block that covers the maximum number of currently
      uncovered t-subsets. Among ties, prefer the one that covers the most
      'rare' t-subsets (those that appear in fewer total k-blocks).
    """
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subs)
    if not uncovered:
        return []
    
    # For efficiency, we'll track uncovered t-subsets in a set
    # and use a greedy approach with scoring.
    
    # To speed up, we can precompute for each point the list of t-subsets containing it,
    # but that's memory intensive. Instead, we'll use a smarter greedy.
    
    # Alternative approach: use a randomized greedy with local search.
    # But first, let's try a more effective greedy:
    # 1. Start with an empty set of blocks.
    # 2. While there are uncovered t-subsets:
    #    a. Find the k-block that covers the most uncovered t-subsets.
    #    b. Add it to the solution.
    #    c. Mark all t-subsets in this block as covered.
    
    # To make this faster and better, we use a heuristic:
    # - For each candidate block, compute the number of uncovered t-subsets it covers.
    # - Use a priority queue or just scan all blocks.
    
    # Since v=19, k=7, the number of possible blocks is C(19,7) = 50388, which is manageable.
    # But we don't need to generate all of them upfront; we can generate candidates on the fly.
    
    # Better approach: use a greedy that builds blocks one at a time,
    # selecting points to maximize coverage of uncovered t-subsets.
    
    # Let's use a two-phase approach:
    # Phase 1: Greedy set cover with scoring.
    # Phase 2: Local search to improve.
    
    # For now, let's implement a strong greedy:
    # - Maintain uncovered as a set of frozensets for O(1) lookup.
    # - For each step, try to find the best k-block.
    
    uncovered_set = set(frozenset(s) for s in uncovered)
    blocks = []
    
    # Precompute: for each point, which t-subsets contain it?
    # This helps in evaluating the marginal gain of adding a point to a block.
    point_to_tsubs = defaultdict(set)
    for ts in all_t_subs:
        for p in ts:
            point_to_tsubs[p].add(frozenset(ts))
    
    # Greedy construction of each block:
    while uncovered_set:
        # Start a new block with the point that appears in the most uncovered t-subsets
        # or with a point from the smallest uncovered t-subset (to ensure progress).
        
        # Find the uncovered t-subset with the smallest "potential" (i.e., fewest possible blocks)
        # or just pick the first one.
        # Let's pick the uncovered t-subset that has the minimum number of points
        # that could extend it to a k-block while covering many uncovered t-subsets.
        
        # Simpler: start with a point that is in many uncovered t-subsets.
        point_counts = {p: len(point_to_tsubs[p] & uncovered_set) for p in range(v)}
        start_point = max(range(v), key=lambda p: point_counts[p])
        
        current_block = [start_point]
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(current_block) < k:
            best_point = None
            best_gain = -1
            for p in range(v):
                if p in current_block:
                    continue
                # Compute how many new uncovered t-subsets would be covered by adding p
                # A t-subset is covered by current_block + [p] if it's a subset of current_block + [p]
                # and it contains p (since it wasn't covered by current_block alone if t > len(current_block))
                
                # More precisely: a t-subset s is newly covered if s ⊆ current_block ∪ {p} and p ∈ s
                # and s was uncovered before.
                
                # For efficiency, iterate over t-subsets containing p that are subsets of current_block ∪ {p}
                # i.e., s \ {p} ⊆ current_block
                gain = 0
                for ts in point_to_tsubs[p]:
                    if ts in uncovered_set:
                        # Check if ts \ {p} ⊆ current_block
                        if all(x in current_block for x in ts if x != p):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            current_block.append(best_point)
        
        current_block.sort()
        blocks.append(current_block)
        
        # Mark all t-subsets in this block as covered
        for ts in itertools.combinations(current_block, t):
            uncovered_set.discard(frozenset(ts))
    
    return blocks
# EVOLVE-BLOCK-END
