"""Covering design C(24,7,3) - optimized greedy with local search and time budget."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 138.0  # safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def get_block_covered(block):
        """Get the set of t-subset indices covered by a block."""
        return frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(block, t))
    
    def greedy_cover(seed, order_mode="random"):
        """Greedy covering with a given random seed and ordering mode."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        if not uncovered:
            return blocks
        
        # Choose initial block from an uncovered t-subset
        if order_mode == "random":
            start_idx = rng.choice(list(uncovered))
        else:
            start_idx = min(uncovered)
        
        block = list(all_t_subsets[start_idx])
        
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
                    if t_subset_to_idx.get(ts) in uncovered:
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
            uncovered.discard(t_subset_to_idx[ts])
        
        # Check if we're done
        if not uncovered:
            return blocks
        
        # Continue with remaining uncovered
        while uncovered:
            if time.time() > deadline - 5.0:
                return blocks
            
            if order_mode == "random":
                start_idx = rng.choice(list(uncovered))
            else:
                start_idx = min(uncovered)
            
            block = list(all_t_subsets[start_idx])
            
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
                        if t_subset_to_idx.get(ts) in uncovered:
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
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks
    
    def local_search_remove(blocks, seed):
        """Try to remove blocks from a covering while maintaining validity."""
        rng = random.Random(seed)
        blocks = list(blocks)
        n = len(blocks)
        
        # Precompute coverage for each block
        block_covers = [get_block_covered(b) for b in blocks]
        
        # Count how many times each t-subset is covered
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bc in block_covers:
            for idx in bc:
                cover_count[idx] += 1
        
        # Try to remove blocks in random order
        indices = list(range(n))
        rng.shuffle(indices)
        
        removed = []
        for i in indices:
            if i in removed:
                continue
            if time.time() > deadline - 5.0:
                break
            
            # Check if removing block i keeps all t-subsets covered
            can_remove = True
            for idx in block_covers[i]:
                if cover_count[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block i
                for idx in block_covers[i]:
                    cover_count[idx] -= 1
                removed.append(i)
        
        if removed:
            return [b for i, b in enumerate(blocks) if i not in removed]
        return blocks
    
    # Strategy 1: Standard greedy (deterministic)
    best_blocks = None
    best_count = float('inf')
    
    # First, try the original greedy approach
    blocks = greedy_cover(0, "deterministic")
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = blocks
    
    # Try multiple random seeds with local search
    num_attempts = 30
    for i in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_cover(i + 1, "random")
        
        # Apply local search to try to remove redundant blocks
        blocks = local_search_remove(blocks, i + 100)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Try a more sophisticated approach: greedy with lookahead
    def greedy_with_lookahead(seed):
        """Greedy with a small lookahead to choose better starting points."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
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
            
            for start_idx in candidates[:num_to_try]:
                # Simulate how many blocks this starting point would lead to
                # (approximate by checking how many other uncovered t-subsets share points)
                start_ts = all_t_subsets[start_idx]
                shared = 0
                for other_idx in list(uncovered)[:50]:
                    if set(start_ts) & set(all_t_subsets[other_idx]):
                        shared += 1
                # Prefer starts that share many points with other uncovered sets
                score = -shared
                if score < best_score:
                    best_score = score
                    best_start = start_idx
            
            if best_start is None:
                best_start = candidates[0]
            
            block = list(all_t_subsets[best_start])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
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
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks
    
    # Try lookahead greedy with local search
    for i in range(15):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_with_lookahead(i)
        blocks = local_search_remove(blocks, i + 200)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Try a different strategy: start with a large random block and greedily add
    def random_greedy(seed):
        """Start with random blocks and greedily fill them."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
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
            start_idx = rng.choice(list(uncovered))
            block = list(all_t_subsets[start_idx])
            
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
                        if t_subset_to_idx.get(ts) in uncovered:
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
                uncovered.discard(t_subset_to_idx[ts])
        
        # Continue with standard greedy for remaining
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            start_idx = min(uncovered)
            block = list(all_t_subsets[start_idx])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
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
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks
    
    for i in range(10):
        if time.time() > deadline - 10.0:
            break
        blocks = random_greedy(i)
        blocks = local_search_remove(blocks, i + 300)
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
            covered.add(t_subset_to_idx[ts])
    
    if covered != set(range(n_t)):
        # If invalid, fall back to a simple greedy that's guaranteed to work
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            start_idx = min(uncovered)
            block = list(all_t_subsets[start_idx])
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
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
                uncovered.discard(t_subset_to_idx[ts])
        return blocks
    
    return best_blocks
