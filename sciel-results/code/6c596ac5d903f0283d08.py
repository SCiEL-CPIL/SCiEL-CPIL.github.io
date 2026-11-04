import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    best_blocks = None
    best_len = float('inf')
    
    def time_left():
        return deadline - time.time() > 3.0

    def greedy_cover(seed, strategy):
        """Run one greedy covering with given seed and strategy variant."""
        rng = random.Random(seed)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered and time_left():
            # Choose starting t-subset based on strategy
            if strategy == 0:
                # Pick random uncovered
                start = rng.choice(tuple(uncovered))
            elif strategy == 1:
                # Pick uncovered t-subset with rarest point (sample-based)
                sample = list(uncovered)[:300]
                point_counts = defaultdict(int)
                for sub in sample:
                    for p in sub:
                        point_counts[p] += 1
                best_sub = None
                best_rarity = float('inf')
                for sub in sample:
                    rarity = min(point_counts[p] for p in sub)
                    if rarity < best_rarity:
                        best_rarity = rarity
                        best_sub = sub
                start = best_sub if best_sub else rng.choice(tuple(uncovered))
            else:
                # strategy 2: pick lexicographically smallest
                start = min(uncovered)
            
            block = list(start)
            block_set = set(block)
            
            while len(block) < k and time_left():
                best_gain = -1
                best_point = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0 and rng.random() < 0.3:
                        best_point = x
                
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        return blocks, uncovered

    def complete_cover(blocks, uncovered):
        """Complete the covering if time ran out, ignoring deadline."""
        if not uncovered:
            return blocks
        block_set_list = [set(b) for b in blocks]
        while uncovered:
            anchor = min(uncovered)
            block = list(anchor)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = candidates[0]
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            if block not in blocks:
                blocks.append(block)
                block_set_list.append(set(block))
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
        return blocks

    def remove_redundant(blocks):
        """Remove blocks that are not the unique cover for any t-subset."""
        if len(blocks) <= 1:
            return blocks
        
        # Map each t-subset to the set of block indices that cover it
        tsub_to_blocks = defaultdict(set)
        for i, block in enumerate(blocks):
            for comb in itertools.combinations(block, t):
                tsub_to_blocks[comb].add(i)
        
        # A block can be removed if no t-subset is covered only by it
        removable = set(range(len(blocks)))
        for comb, bset in tsub_to_blocks.items():
            if len(bset) == 1:
                removable.discard(bset.pop())
        
        if not removable:
            return blocks
        
        # Remove in reverse order to preserve indices
        new_blocks = [b for i, b in enumerate(blocks) if i not in removable]
        return new_blocks

    # Try multiple strategies and seeds
    strategies = [0, 1, 2]
    seed_base = 42
    results = []
    
    for strategy in strategies:
        for trial in range(40):
            if not time_left():
                break
            seed = seed_base + strategy * 1000 + trial
            blocks, remaining = greedy_cover(seed, strategy)
            
            if not blocks:
                continue
            
            # Complete if needed
            if remaining:
                blocks = complete_cover(blocks, remaining)
            
            # Local search: remove redundant blocks
            if time_left():
                blocks = remove_redundant(blocks)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                results.append((len(blocks), strategy, trial))
            
            if not time_left():
                break
    
    if best_blocks is None:
        # Final fallback: simple deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = candidates[0]
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
