import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 145.0  # safety margin before 150s
    best_blocks = None
    best_len = float('inf')
    
    def time_left():
        return deadline - time.time() > 2.0
    
    def greedy_cover(seed, strategy):
        """Run one greedy covering with given seed and strategy variant."""
        rng = random.Random(seed)
        
        # uncovered: dict from frozenset -> True, or use a set of tuples
        # For efficiency, use a set of sorted tuples
        all_tsubs = set(itertools.combinations(range(v), t))
        uncovered = set(all_tsubs)
        blocks = []
        
        while uncovered and time_left():
            # Choose starting t-subset
            if strategy == 0:
                # Pick random uncovered
                start = rng.choice(list(uncovered))
            elif strategy == 1:
                # Pick the uncovered t-subset with most "hard" points
                # (points that appear in few uncovered t-subsets)
                point_counts = defaultdict(int)
                for sub in list(uncovered)[:200]:  # sample for speed
                    for p in sub:
                        point_counts[p] += 1
                # Pick sub containing the rarest point
                best_sub = None
                best_rarity = float('inf')
                for sub in list(uncovered)[:200]:
                    rarity = min(point_counts[p] for p in sub)
                    if rarity < best_rarity:
                        best_rarity = rarity
                        best_sub = sub
                start = best_sub if best_sub else rng.choice(list(uncovered))
            else:
                # strategy 2: pick min uncovered (deterministic-ish)
                start = min(uncovered)
            
            block = list(start)
            
            while len(block) < k and time_left():
                # Compute gain for each candidate point
                best_gain = -1
                best_point = -1
                best_tie = None
                
                for x in range(v):
                    if x in block:
                        continue
                    # Count uncovered t-subsets covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    
                    if gain > best_gain or (gain == best_gain and (best_tie is None or rng.random() < 0.5)):
                        best_gain = gain
                        best_point = x
                        best_tie = True
                
                if best_point == -1:
                    # Fallback: pick random
                    candidates = [x for x in range(v) if x not in block]
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Remove all covered t-subsets
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        return blocks
    
    # Try multiple strategies and seeds
    strategies = [0, 1, 2]
    seed_base = 42
    
    for strategy in strategies:
        for trial in range(50):
            if not time_left():
                break
            seed = seed_base + strategy * 1000 + trial
            blocks = greedy_cover(seed, strategy)
            if blocks and len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
            if not time_left():
                break
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
