# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict
import sys

sys.setrecursionlimit(10000)


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using greedy construction followed by
    simulated annealing with multi-move neighborhood (remove/swap).
    
    Strategy:
    1. Greedy initialization: repeatedly pick the k-set that covers the most uncovered t-subsets.
    2. Simulated annealing: try removing a block (if it doesn't break coverage) or swapping 
       one point in a block with another point to reduce total blocks.
    3. Deadline guard: return best valid covering before 115s.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, safety margin before 120s
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Helper: get all t-subsets within a k-set (block)
    def block_tsubsets(block):
        return [tuple(block[i:i+1]) for i in range(len(block))]  # placeholder
    
    # Better: precompute for a given sorted k-tuple
    def get_tsubs_from_block(block):
        return list(itertools.combinations(block, t))
    
    # Greedy construction
    def greedy_construct():
        uncovered = set(all_tsubsets)
        blocks = []
        
        # Precompute for each point, which t-subsets it participates in
        point_tsubs = defaultdict(list)
        for ts in all_tsubsets:
            for p in ts:
                point_tsubs[p].append(ts)
        
        while uncovered:
            # Find the k-set that covers the most uncovered t-subsets
            # Use a heuristic: start with the first uncovered t-subset's elements,
            # then greedily add points that maximize coverage
            first_ts = min(uncovered)
            block = list(first_ts)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for p in range(v):
                    if p in block:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding p
                    gain = 0
                    for ts in point_tsubs[p]:
                        if ts in uncovered and all(x in block or x == p for x in ts):
                            # This t-subset contains p and all other elements are in block
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point >= 0:
                    block.append(best_point)
                else:
                    # Fallback: add any unused point
                    for p in range(v):
                        if p not in block:
                            block.append(p)
                            break
            
            block.sort()
            blocks.append(tuple(block))
            
            # Remove covered t-subsets
            for ts in get_tsubs_from_block(block):
                uncovered.discard(ts)
        
        return [list(b) for b in blocks]
    
    # Check if a set of blocks covers all t-subsets
    def is_valid_covering(blocks):
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        return len(covered) == n_tsubsets
    
    # Count coverage: for each t-subset, how many blocks cover it
    def compute_coverage(blocks):
        coverage = [0] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    coverage[idx] += 1
        return coverage
    
    # Try to remove block i if coverage remains valid
    def can_remove_block(blocks, i, coverage):
        # Check if any t-subset in block i has coverage == 1
        block = blocks[i]
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None and coverage[idx] <= 1:
                return False
        return True
    
    # Simulated annealing
    def simulated_annealing(initial_blocks, max_time=110.0):
        blocks = [list(b) for b in initial_blocks]
        coverage = compute_coverage(blocks)
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        # Temperature schedule
        T = 5.0
        cooling = 0.9995
        min_T = 0.1
        
        step = 0
        while time.time() - start_time < max_time:
            if T < min_T:
                T = min_T
            
            step += 1
            n_blocks = len(blocks)
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3 and n_blocks > 1:
                # Try to remove a random block
                i = random.randint(0, n_blocks - 1)
                if can_remove_block(blocks, i, coverage):
                    # Compute delta: removing block i reduces block count by 1
                    # Update coverage
                    block = blocks[i]
                    for ts in itertools.combinations(block, t):
                        idx = tsub_index.get(ts)
                        if idx is not None:
                            coverage[idx] -= 1
                    
                    # Check validity
                    valid = all(c >= 1 for c in coverage)
                    if valid:
                        blocks.pop(i)
                        best_blocks = [list(b) for b in blocks]
                        best_len = len(blocks)
                    else:
                        # Roll back
                        for ts in itertools.combinations(block, t):
                            idx = tsub_index.get(ts)
                            if idx is not None:
                                coverage[idx] += 1
                        blocks.insert(i, block)
                        
            elif move_type < 0.7 and n_blocks > 0:
                # Try to swap a point in a random block with another point
                i = random.randint(0, n_blocks - 1)
                block = blocks[i]
                if len(block) == 0:
                    continue
                j = random.randint(0, len(block) - 1)
                old_point = block[j]
                
                # Find a new point not already in block
                candidates = [p for p in range(v) if p not in block]
                if not candidates:
                    continue
                new_point = random.choice(candidates)
                
                # Compute effect on coverage
                # Remove old point from block, add new point
                old_block = tuple(sorted(block))
                new_block_list = block[:]
                new_block_list[j] = new_point
                new_block_list.sort()
                new_block = tuple(new_block_list)
                
                # Coverage change: subtract old block's t-subsets, add new block's t-subsets
                delta_coverage = {}
                for ts in itertools.combinations(old_block, t):
                    idx = tsub_index.get(ts)
                    if idx is not None:
                        delta_coverage[idx] = delta_coverage.get(idx, 0) - 1
                for ts in itertools.combinations(new_block, t):
                    idx = tsub_index.get(ts)
                    if idx is not None:
                        delta_coverage[idx] = delta_coverage.get(idx, 0) + 1
                
                # Check if any coverage would go to 0
                valid = True
                for idx, delta in delta_coverage.items():
                    if coverage[idx] + delta < 1:
                        valid = False
                        break
                
                if valid:
                    # Apply the swap
                    for idx, delta in delta_coverage.items():
                        coverage[idx] += delta
                    blocks[i] = list(new_block)
                    
                    # This move doesn't change block count, but may enable future removals
                    # Accept with some probability (it's neutral in block count)
                    if random.random() < 0.5:
                        pass  # Keep it
                    else:
                        # Roll back
                        for idx, delta in delta_coverage.items():
                            coverage[idx] -= delta
                        blocks[i] = list(old_block)
                # else: reject
                
            else:
                # Try to add a new block that covers many uncovered or sparsely covered t-subsets
                # This is less useful but can help escape local optima
                if n_blocks < 500:  # Don't grow too large
                    # Find t-subsets with lowest coverage
                    min_cov = min(coverage)
                    if min_cov <= 2:
                        # Pick a random t-subset with low coverage and build a block around it
                        low_cov_tsubs = [ts for ts in all_tsubsets if coverage[tsub_index[ts]] <= 2]
                        if low_cov_tsubs:
                            ts = random.choice(low_cov_tsubs)
                            block = list(ts)
                            while len(block) < k:
                                best_p = -1
                                best_gain = 0
                                for p in range(v):
                                    if p in block:
                                        continue
                                    gain = 0
                                    for t2 in itertools.combinations(block, t-1):
                                        ts2 = tuple(sorted(t2 + (p,)))
                                        if ts2 in tsub_index:
                                            gain += 1 / (coverage[tsub_index[ts2]] + 1)
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_p = p
                                if best_p >= 0:
                                    block.append(best_p)
                                else:
                                    for p in range(v):
                                        if p not in block:
                                            block.append(p)
                                            break
                            block.sort()
                            block_t = tuple(block)
                            
                            # Compute coverage delta
                            delta_cov = {}
                            for t2 in itertools.combinations(block_t, t):
                                idx = tsub_index.get(t2)
                                if idx is not None:
                                    delta_cov[idx] = delta_cov.get(idx, 0) + 1
                            
                            # Adding a block increases block count by 1, but may allow future removals
                            # Only add if it significantly improves coverage of low-coverage areas
                            improvement = sum(1 for idx, d in delta_cov.items() if coverage[idx] <= 2)
                            if improvement >= 3:
                                for idx, d in delta_cov.items():
                                    coverage[idx] += d
                                blocks.append(list(block_t))
            
            # Cooling
            T *= cooling
            
            # Periodic restart: if stuck, try a small perturbation
            if step % 500 == 0:
                # Perturb: randomly swap a point in a random block
                if blocks:
                    i = random.randint(0, len(blocks) - 1)
                    block = blocks[i]
                    if block:
                        j = random.randint(0, len(block) - 1)
                        candidates = [p for p in range(v) if p not in block]
                        if candidates:
                            new_p = random.choice(candidates)
                            old_p = block[j]
                            block[j] = new_p
                            block.sort()
                            # Recompute coverage for this block
                            old_block_t = tuple(sorted(block))
                            # This is a rough perturbation; recompute full coverage
                            coverage = compute_coverage(blocks)
        
        return best_blocks, best_len
    
    # Main algorithm
    # Step 1: Greedy construction
    blocks = greedy_construct()
    best_blocks = [list(b) for b in blocks]
    best_len = len(blocks)
    
    # Step 2: Simulated annealing
    if time.time() - start_time < DEADLINE:
        sa_blocks, sa_len = simulated_annealing(blocks, max_time=min(110.0, DEADLINE - (time.time() - start_time)))
        if sa_len < best_len:
            best_blocks = sa_blocks
            best_len = sa_len
    
    # Final cleanup: try to remove any redundant blocks
    if time.time() - start_time < DEADLINE - 5:
        coverage = compute_coverage(best_blocks)
        changed = True
        while changed and time.time() - start_time < DEADLINE - 5:
            changed = False
            for i in range(len(best_blocks)):
                if can_remove_block(best_blocks, i, coverage):
                    block = best_blocks[i]
                    for ts in itertools.combinations(block, t):
                        idx = tsub_index.get(ts)
                        if idx is not None:
                            coverage[idx] -= 1
                    best_blocks.pop(i)
                    coverage = compute_coverage(best_blocks)
                    changed = True
                    break
    
    return best_blocks
# EVOLVE-BLOCK-END
