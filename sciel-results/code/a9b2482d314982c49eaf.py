"""Covering design C(24,7,3) - optimized greedy with multiple strategies and time budget."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each block (k-subset), precompute which t-subsets it covers
    # But C(24,7) = 346104, which is large. Instead, we compute on the fly.
    
    def count_covered(block, uncovered_set):
        """Count how many uncovered t-subsets are covered by adding a point to block."""
        if len(block) < t - 1:
            return 0
        count = 0
        for rest in itertools.combinations(block, t - 1):
            # rest is a (t-1)-tuple, add x to make t-subset
            for x in range(v):
                if x not in block:
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered_set:
                        count += 1
        return count
    
    def greedy_cover(seed, order_mode="random"):
        """Greedy covering with a given random seed and ordering mode."""
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        if not uncovered:
            return blocks
        
        # Choose initial block from an uncovered t-subset
        if order_mode == "random":
            start_ts = rng.choice(list(uncovered))
        else:
            start_ts = min(uncovered)
        
        block = list(start_ts)
        
        while len(block) < k:
            # Find the point that covers the most uncovered t-subsets
            best_x = None
            best_gain = -1
            
            candidates = [x for x in range(v) if x not in block]
            if order_mode == "random":
                rng.shuffle(candidates)
            
            for x in candidates:
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
        
        # Check if we're done
        if not uncovered:
            return blocks
        
        # Continue with remaining uncovered
        while uncovered:
            if time.time() > deadline - 5.0:
                return blocks
            
            if order_mode == "random":
                start_ts = rng.choice(list(uncovered))
            else:
                start_ts = min(uncovered)
            
            block = list(start_ts)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                candidates = [x for x in range(v) if x not in block]
                if order_mode == "random":
                    rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        
        return blocks
    
    # Strategy 1: Standard greedy (deterministic)
    best_blocks = None
    best_count = float('inf')
    
    # First, try the original greedy approach
    blocks = greedy_cover(0, "deterministic")
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = blocks
    
    # Try multiple random seeds
    num_attempts = 20
    for i in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_cover(i + 1, "random")
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Try a more sophisticated approach: greedy with lookahead
    def greedy_with_lookahead(seed):
        """Greedy with a small lookahead to choose better starting points."""
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        if not uncovered:
            return blocks
        
        while uncovered:
            if time.time() > deadline - 5.0:
                return blocks
            
            # Try a few candidate starting t-subsets and pick the best
            candidates = list(uncovered)
            rng.shuffle(candidates)
            num_to_try = min(5, len(candidates))
            
            best_start = None
            best_score = float('inf')
            
            for start_ts in candidates[:num_to_try]:
                # Simulate how many blocks this starting point would lead to
                # (approximate by checking how many other uncovered t-subsets share points)
                shared = 0
                for other in list(uncovered)[:50]:
                    if set(start_ts) & set(other):
                        shared += 1
                # Prefer starts that share many points with other uncovered sets
                score = -shared
                if score < best_score:
                    best_score = score
                    best_start = start_ts
            
            if best_start is None:
                best_start = candidates[0]
            
            block = list(best_start)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        
        return blocks
    
    # Try lookahead greedy
    for i in range(10):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_with_lookahead(i)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Try a different strategy: start with a large random block and greedily add
    def random_greedy(seed):
        """Start with random blocks and greedily fill them."""
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        if not uncovered:
            return blocks
        
        # First, create a few random blocks
        num_initial = 3
        for _ in range(num_initial):
            if not uncovered:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random t-subset from uncovered as seed
            start_ts = rng.choice(list(uncovered))
            block = list(start_ts)
            
            # Greedily add points
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        
        # Continue with standard greedy for remaining
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            start_ts = min(uncovered)
            block = list(start_ts)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        
        return blocks
    
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        blocks = random_greedy(i)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Final check: ensure we have a valid covering
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    # Verify the covering is valid
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    
    if covered != set(all_t_subsets):
        # If invalid, fall back to a simple greedy that's guaranteed to work
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            start_ts = min(uncovered)
            block = list(start_ts)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        return blocks
    
    return best_blocks
