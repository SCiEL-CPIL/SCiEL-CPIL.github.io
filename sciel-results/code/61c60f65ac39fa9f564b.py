<<<<<<< SEARCH
"""Simulated annealing for covering design C(24,7,3) with numpy-accelerated coverage tracking."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks
=======
"""Simulated annealing for covering design C(24,7,3) with numpy-accelerated coverage tracking."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    block_set = set()
    
    while uncovered:
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = tuple(sorted(rest + (x,)))
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(c)))
                
    return blocks
>>>>>>> REPLACE

