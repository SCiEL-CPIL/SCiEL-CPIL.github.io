"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy with multiple restarts."""
    deadline = time.time() + 115  # safety margin before 120s limit
    
    def _generate_greedy(v, k, t, rng):
        """Generate a covering using randomized greedy algorithm."""
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_set = rng.choice(list(uncovered))
            block = list(start_set)
            
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point is not None and x < best_point:
                        # Tie-breaking: prefer smaller point for determinism in some cases
                        # But we want randomness, so let's use rng for ties
                        if rng.random() < 0.5:
                            best_point = x
                
                if best_point is None:
                    break
                block.append(best_point)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                uncovered.difference_update(itertools.combinations(block, t))
            
            # Check deadline
            if time.time() > deadline:
                break
        
        return blocks if not uncovered else None
    
    def _verify(blocks, v, t):
        """Verify that all t-subsets are covered."""
        covered = set()
        for block in blocks:
            covered.update(itertools.combinations(block, t))
        return len(covered) == len(list(itertools.combinations(range(v), t)))
    
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple random restarts
    seed = 42
    for attempt in range(100):  # Max attempts, but limited by time
        if time.time() > deadline:
            break
        
        rng = random.Random(seed + attempt)
        blocks = _generate_greedy(v, k, t, rng)
        
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
            # If we found a good solution, we can stop early if time allows
            if best_count <= 151:  # Known upper bound
                break
    
    # If we didn't find a valid solution, use a deterministic fallback
    if best_blocks is None or not _verify(best_blocks, v, t):
        # Fallback to original greedy approach (deterministic)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
