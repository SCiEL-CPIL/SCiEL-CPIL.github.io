# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    
    # For C(24,7,3): total_t = 2024
    
    # We'll use a greedy approach with a better scoring function.
    # Key insight: when building a block, we want to maximize coverage of
    # "hard" t-subsets (those that are covered by few existing blocks).
    
    # Track coverage count for each t-subset
    # Index t-subsets for fast lookup
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # coverage[i] = number of blocks that cover all_t_subsets[i]
    coverage = [0] * total_t
    
    # uncovered[i] = True if all_t_subsets[i] is not yet covered
    uncovered = [True] * total_t
    uncovered_count = total_t
    
    blocks = []
    
    # Helper: get list of uncovered t-subsets (as indices)
    # We'll maintain this incrementally
    
    # For efficient gain computation, precompute for each point x and each (t-1)-subset,
    # which t-subsets contain both
    # Actually, let's think differently.
    
    # When we have a partial block B, adding point x covers all t-subsets of the form
    # {x} ∪ S where S is a (t-1)-subset of B.
    # So the gain of adding x to B is the number of uncovered t-subsets of that form.
    
    # Precompute: for each (t-1)-subset S and point x not in S, the t-subset S ∪ {x}
    # and its index.
    
    # For t=3: (t-1)=2, so for each pair (a,b) and point x not in {a,b}, 
    # the triple is (a,b,x) sorted.
    
    # Let's precompute a mapping: for each pair (a,b) with a<b, and each x not in {a,b},
    # store the index of the triple sorted((a,b,x)).
    
    # Actually, for general t, we need for each (t-1)-subset S and point x not in S,
    # the index of sorted(S ∪ {x}).
    
    # For efficiency with t=3, v=24, k=7:
    # Number of pairs: C(24,2) = 276
    # For each pair, 22 possible x's
    # Total: 276 * 22 = 6072 entries
    
    # Let's precompute this for t=3 specifically, but make it general enough.
    
    # General approach: precompute for each (t-1)-subset S (as a frozenset or tuple),
    # a dict mapping x -> index of sorted(S ∪ {x})
    
    # For t=3, (t-1)-subsets are pairs.
    # Let's just do it for the general case but optimize for the expected parameters.
    
    if t == 3:
        # Precompute triple_index[pair][x] = index of triple
        # pair is (a,b) with a < b
        triple_index = {}
        for a in range(v):
            for b in range(a+1, v):
                pair = (a, b)
                triple_index[pair] = {}
                for x in range(v):
                    if x != a and x != b:
                        triple = tuple(sorted((a, b, x)))
                        triple_index[pair][x] = t_subset_index[triple]
    else:
        # General case: precompute for each (t-1)-subset
        triple_index = None
    
    def get_gain(block, x, t=3):
        """Compute how many uncovered t-subsets are covered by adding x to block."""
        if t == 3:
            # block has some points; adding x covers triples {x, a, b} for all pairs (a,b) in block
            gain = 0
            for i in range(len(block)):
                for j in range(i+1, len(block)):
                    a, b = block[i], block[j]
                    if a > b:
                        a, b = b, a
                    pair = (a, b)
                    idx = triple_index[pair][x]
                    if uncovered[idx]:
                        gain += 1
            return gain
        else:
            # General case
            gain = 0
            for rest in itertools.combinations(block, t-1):
                ts = tuple(sorted(rest + (x,)))
                idx = t_subset_index.get(ts, -1)
                if idx >= 0 and uncovered[idx]:
                    gain += 1
            return gain
    
    def cover_block(block):
        """Update coverage and uncovered arrays for the given block."""
        nonlocal uncovered_count
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
            coverage[idx] += 1
    
    # Greedy construction with weighting
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find the "hardest" uncovered t-subset to start a new block
        # Use the one with minimum coverage (all are 0 initially, so just pick first)
        # Actually, all uncovered have coverage 0 by definition.
        # Let's pick one that's "central" or just the first one.
        
        # Find first uncovered
        start_ts = None
        for i in range(total_t):
            if uncovered[i]:
                start_ts = all_t_subsets[i]
                break
        
        if start_ts is None:
            break
        
        # Start block with the points of start_ts
        block = list(start_ts)
        
        # Greedily add points to maximize gain
        while len(block) < k:
            if time.time() > deadline:
                break
            
            # Find best point to add
            best_x = None
            best_gain = -1
            for x in range(v):
                if x not in block:
                    g = get_gain(block, x, t)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            
            if best_x is None or best_gain <= 0:
                break
            
            block.append(best_x)
        
        if len(block) >= t:
            block.sort()
            blocks.append(block)
            cover_block(block)
    
    return blocks
# EVOLVE-BLOCK-END
