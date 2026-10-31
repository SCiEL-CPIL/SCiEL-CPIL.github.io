"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # leave margin under 120s
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    # This is expensive for large v, but for v=19, k=7, t=4 it's manageable
    # C(19,7) = 50388, each covers C(7,4)=35 t-subsets
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        
        while n_uncovered > 0:
            # Find the first uncovered t-subset
            first_uncovered = None
            for i in range(n_tsubsets):
                if uncovered[i]:
                    first_uncovered = i
                    break
            if first_uncovered is None:
                break
            
            start_ts = all_tsubsets[first_uncovered]
            block = list(start_ts)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                block_set = set(block)
                best_point = -1
                best_gain = -1
                for x in all_points:
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_index.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            
            block.sort()
            block_set = set(block)
            # Mark all t-subsets within this block as covered
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[ts]
                if uncovered[idx]:
                    uncovered[idx] = False
                    n_uncovered -= 1
            blocks.append(tuple(block))
        
        return blocks, uncovered, n_uncovered
    
    def is_redundant(blocks, block_idx, coverage_count):
        """Check if block at block_idx is redundant (all its t-subsets covered by other blocks)."""
        block = blocks[block_idx]
        for ts in itertools.combinations(block, t):
            idx = tsubset_index[ts]
            if coverage_count[idx] <= 1:
                return False
        return True
    
    def local_search(blocks, uncovered, n_uncovered, max_time):
        """Try to improve the covering by removing redundant blocks and replacing blocks."""
        blocks = list(blocks)
        uncovered = list(uncovered)
        n_uncovered = n_uncovered
        
        # Build coverage count: for each t-subset, how many blocks cover it
        coverage_count = [0] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[ts]
                coverage_count[idx] += 1
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Pass 1: Remove redundant blocks
            i = 0
            while i < len(blocks):
                if is_redundant(blocks, i, coverage_count):
                    block = blocks[i]
                    for ts in itertools.combinations(block, t):
                        idx = tsubset_index[ts]
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    improved = True
                else:
                    i += 1
            
            # Pass 2: Try to replace each block with a better one
            for i in range(len(blocks)):
                if (time.time() - start_time) > max_time:
                    break
                old_block = blocks[i]
                old_block_set = set(old_block)
                
                # Count uncovered t-subsets covered by old block
                old_uncovered_covered = 0
                for ts in itertools.combinations(old_block, t):
                    idx = tsubset_index[ts]
                    if not uncovered[idx]:
                        old_uncovered_covered += 1
                
                # Try a few random candidate blocks
                best_candidate = None
                best_new_uncovered_covered = old_uncovered_covered
                
                # Generate random candidates
                for _ in range(50):
                    candidate = random.sample(all_points, k)
                    candidate.sort()
                    candidate_set = set(candidate)
                    
                    # Count how many uncovered t-subsets this candidate covers
                    new_uncovered_covered = 0
                    for ts in itertools.combinations(candidate, t):
                        idx = tsubset_index[ts]
                        if not uncovered[idx]:
                            new_uncovered_covered += 1
                    
                    if new_uncovered_covered > best_new_uncovered_covered:
                        best_new_uncovered_covered = new_uncovered_covered
                        best_candidate = candidate
                
                if best_candidate is not None:
                    # Update coverage
                    for ts in itertools.combinations(old_block, t):
                        idx = tsubset_index[ts]
                        coverage_count[idx] -= 1
                    for ts in itertools.combinations(best_candidate, t):
                        idx = tsubset_index[ts]
                        coverage_count[idx] += 1
                    blocks[i] = tuple(best_candidate)
                    improved = True
        
        # Recompute uncovered
        uncovered = [True] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[ts]
                uncovered[idx] = False
        n_uncovered = sum(uncovered)
        
        return blocks, uncovered, n_uncovered
    
    best_blocks = None
    best_n_uncovered = n_tsubsets + 1
    
    # Try multiple random seeds for greedy initialization
    n_restarts = 5
    for seed in range(n_restarts):
        if (time.time() - start_time) > time_limit:
            break
        
        blocks, uncovered, n_uncovered = greedy_construct(seed)
        
        if n_uncovered == 0:
            # Valid covering, try to improve
            blocks, uncovered, n_uncovered = local_search(blocks, uncovered, n_uncovered, time_limit)
        
        if n_uncovered == 0 and len(blocks) < best_n_uncovered:
            best_n_uncovered = len(blocks)
            best_blocks = blocks
    
    # If we didn't find a valid covering, fall back to greedy
    if best_blocks is None:
        blocks, uncovered, n_uncovered = greedy_construct(0)
        best_blocks = blocks
    
    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
