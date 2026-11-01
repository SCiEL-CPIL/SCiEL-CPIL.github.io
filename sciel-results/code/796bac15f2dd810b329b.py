"""Covering design C(24,7,3): Fast greedy with randomized restarts and efficient local search."""
import itertools
import time
import random
from collections import defaultdict

def _fast_greedy(v, k, t, rng, deadline):
    """
    Fast greedy construction.
    At each step, pick the block that covers the most currently uncovered t-subsets.
    """
    # Pre-compute all t-subsets
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsets)
    blocks = []
    
    # To speed up, we can't just iterate all C(24,7) blocks each step (245k blocks * ~76 steps = 18M checks, might be slow in Python if not optimized).
    # Instead, use a randomized heuristic: 
    # 1. Pick a random uncovered t-set.
    # 2. Find all blocks containing it.
    # 3. Pick the one from that set that covers the most OTHER uncovered t-sets.
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered triple to anchor the search
        # This reduces the search space significantly
        anchor = rng.choice(list(uncovered))
        
        # Find candidate blocks containing 'anchor'
        # Instead of generating all blocks, we can generate blocks containing the anchor's elements.
        # The anchor is a set of t elements. We need to add k-t elements from the remaining v-t elements.
        anchor_list = list(anchor)
        remaining_points = [x for x in range(v) if x not in anchor]
        
        # We need to choose k-t points from remaining_points
        # Number of candidates: C(v-t, k-t)
        # For v=24, t=3, k=7: C(21, 4) = 5985 candidates. This is very fast.
        
        best_block = None
        best_gain = -1
        best_block_set = None
        
        # Iterate through all combinations of (k-t) points from remaining
        for combo in itertools.combinations(remaining_points, k - t):
            block_tuple = tuple(sorted(anchor_list + list(combo)))
            block_set = set(block_tuple)
            
            # Calculate gain: how many uncovered t-sets are covered by this block?
            gain = 0
            # The block covers C(k, t) t-sets.
            # We can check which of these are in 'uncovered'
            for tset in itertools.combinations(block_tuple, t):
                if frozenset(tset) in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block_tuple
                best_block_set = block_set
                # Early exit if we cover all possible? No, max gain is C(k,t).
                # But if gain == C(k,t), it's perfect for this step.
                if gain == 35: # C(7,3)
                    break
        
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered
        for tset in itertools.combinations(best_block, t):
            uncovered.discard(frozenset(tset))
            
    return blocks

def _local_search(blocks, v, k, t, deadline):
    """
    Efficient local search:
    1. Remove redundant blocks.
    2. Try to swap a block with a random alternative to see if it enables more removals (optional, risky for time).
    Given time constraints, focus on redundancy removal which is safe and fast.
    """
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Build coverage count
    triple_count = defaultdict(int)
    block_triples = []
    for b in blocks:
        triples = [frozenset(c) for c in itertools.combinations(b, t)]
        block_triples.append(triples)
        for tr in triples:
            triple_count[tr] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Remove redundant blocks
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                return blocks
            redundant = True
            for tr in block_triples[i]:
                if triple_count[tr] <= 1:
                    redundant = False
                    break
            if redundant:
                for tr in block_triples[i]:
                    triple_count[tr] -= 1
                    if triple_count[tr] == 0:
                        del triple_count[tr]
                blocks.pop(i)
                block_triples.pop(i)
                n -= 1
                improved = True
        
        # Pass 2: Simple Swap (Try to replace a block with a random one that covers same criticals)
        # This is expensive, so we limit it.
        # Given the strict time limit, we might skip complex swaps and rely on multiple restarts of greedy.
        # The redundancy removal is the most effective "cleanup" after a greedy.
        
    return blocks

def construct(v, k, t):
    start_time = time.time()
    # 120s limit, leave 10s buffer for safety
    deadline = start_time + 110
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy attempts
    # Each attempt is fast (C(21,4) checks per block, ~76 blocks -> ~450k checks per run, very fast in Python)
    # We can run many restarts.
    
    attempt = 0
    while time.time() < deadline - 5: # Leave 5s for final cleanup/return
        attempt += 1
        rng = random.Random(attempt)
        
        # Greedy phase
        blocks = _fast_greedy(v, k, t, rng, deadline - 10)
        
        if not blocks:
            continue
            
        # Local Search phase (Redundancy removal)
        ls_blocks = _local_search(blocks, v, k, t, deadline - 5)
        
        if len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = [list(b) for b in ls_blocks]
            
        # If we get a very good result (close to lower bound 69), we might want to stop early?
        # But we don't know the exact LB for sure, just 69.
        # If we hit 70, it's great.
        # Let's just keep trying until time runs out.
        
    if best_blocks is None:
        # Fallback: simple greedy if something went wrong
        rng = random.Random(0)
        best_blocks = _fast_greedy(v, k, t, rng, time.time() + 1)
        
    return best_blocks
