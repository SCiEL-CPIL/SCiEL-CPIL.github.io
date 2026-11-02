"""Covering design C(v,k,t) with WalkSAT-style stochastic local search."""
import itertools
import random
import time
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115
    start_time = time.time()
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    def block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def greedy_build(seed, randomize_order=True):
        """Randomized greedy covering."""
        rng = random.Random(seed)
        uncovered_mask = np.ones(n_tsubs, dtype=np.int32)
        blocks = []
        
        tsub_order = np.arange(n_tsubs)
        if randomize_order:
            order_list = tsub_order.tolist()
            rng.shuffle(order_list)
            tsub_order = np.array(order_list)
        
        while uncovered_mask.any():
            first_uncovered = None
            for idx in tsub_order:
                if uncovered_mask[idx]:
                    first_uncovered = idx
                    break
            if first_uncovered is None:
                break
            
            block = list(all_tsubs[first_uncovered])
            
            while len(block) < k:
                in_block = set(block)
                candidates = [x for x in range(v) if x not in in_block]
                
                if not candidates:
                    break
                    
                if randomize_order:
                    rng.shuffle(candidates)
                
                best_gain = -1
                best_x = candidates[0]
                for x in candidates:
                    gain = 0
                    for combo in itertools.combinations(block, t-1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and uncovered_mask[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        if rng.random() < 0.5:
                            best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for idx in block_tsub_indices(block):
                uncovered_mask[idx] = 0
        
        return blocks
    
    def deletion_pass(blocks, deadline):
        """Iteratively remove redundant blocks."""
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks:
            indices = block_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        blocks_list = [list(b) for b in blocks]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n_blocks = len(blocks_list)
            for i in range(n_blocks):
                if time.time() > deadline - 5:
                    break
                critical = 0
                for idx in block_tsubs[i]:
                    if cov_count[idx] == 1:
                        critical += 1
                if critical == 0:
                    for idx in block_tsubs[i]:
                        cov_count[idx] -= 1
                    blocks_list.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    break
        
        return blocks_list
    
    def walksat_optimize(initial_blocks, rng, deadline, max_iters=200000):
        """
        WalkSAT-style local search:
        - State: list of blocks
        - Move: remove one block, add another (1-for-1 swap)
        - Heuristic: prefer removing blocks with few critical t-subs, 
                     prefer adding blocks covering many uncovered t-subs
        - Randomization: with probability p, make a random move
        """
        blocks_list = [list(b) for b in initial_blocks]
        n_blocks = len(blocks_list)
        
        # Build coverage
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks_list:
            indices = block_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        best_blocks = [list(b) for b in blocks_list]
        best_n_blocks = n_blocks
        no_improve = 0
        max_no_improve = 5000
        
        # Precompute candidate pool
        candidate_pool = set()
        for _ in range(min(5000, n_tsubs)):
            cb = tuple(sorted(rng.sample(range(v), k)))
            candidate_pool.add(cb)
        for _ in range(2000):
            ts = rng.choice(all_tsubs)
            remaining = [x for x in range(v) if x not in ts]
            extra = rng.sample(remaining, k - t)
            cb = tuple(sorted(list(ts) + extra))
            candidate_pool.add(cb)
        candidate_pool = list(candidate_pool)
        
        iter_count = 0
        p_random = 0.1  # probability of random move
        
        while iter_count < max_iters and time.time() < deadline - 5:
            iter_count += 1
            
            # Check if we've beaten the best
            if n_blocks < best_n_blocks:
                best_blocks = [list(b) for b in blocks_list]
                best_n_blocks = n_blocks
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                # Restart from best
                blocks_list = [list(b) for b in best_blocks]
                cov_count = np.zeros(n_tsubs, dtype=np.int32)
                block_tsubs = []
                for block in blocks_list:
                    indices = block_tsub_indices(block)
                    block_tsubs.append(indices)
                    for idx in indices:
                        cov_count[idx] += 1
                n_blocks = len(blocks_list)
                no_improve = 0
                max_no_improve = int(max_no_improve * 0.7)
                if max_no_improve < 1000:
                    break
            
            # Choose move
            if rng.random() < p_random:
                # Random move
                if n_blocks > 1 and candidate_pool:
                    bi = rng.randrange(n_blocks)
                    cb = rng.choice(candidate_pool)
                    cb_list = list(cb)
                    
                    # Remove block bi
                    for idx in block_tsubs[bi]:
                        cov_count[idx] -= 1
                    blocks_list.pop(bi)
                    block_tsubs.pop(bi)
                    n_blocks -= 1
                    
                    # Add candidate
                    indices = block_tsub_indices(cb_list)
                    blocks_list.append(cb_list)
                    block_tsubs.append(indices)
                    for idx in indices:
                        cov_count[idx] += 1
                    n_blocks += 1
            else:
                # Heuristic move
                # Score each block by number of critical t-subs it covers
                # (t-subs with cov_count == 1)
                block_critical = np.zeros(n_blocks, dtype=np.int32)
                for bi in range(n_blocks):
                    for idx in block_tsubs[bi]:
                        if cov_count[idx] == 1:
                            block_critical[bi] += 1
                
                # Score each candidate by number of uncovered t-subs it covers
                candidate_scores = np.zeros(len(candidate_pool), dtype=np.int32)
                for ci, cb in enumerate(candidate_pool):
                    cb_list = list(cb)
                    indices = block_tsub_indices(cb_list)
                    for idx in indices:
                        if cov_count[idx] == 0:
                            candidate_scores[ci] += 1
                
                # Prefer removing blocks with fewest critical t-subs
                # and adding candidates with most uncovered coverage
                min_critical = np.min(block_critical)
                candidates_to_remove = np.where(block_critical == min_critical)[0]
                bi = rng.choice(candidates_to_remove)
                
                max_score = np.max(candidate_scores)
                candidates_to_add = np.where(candidate_scores == max_score)[0]
                ci = rng.choice(candidates_to_add)
                
                cb = candidate_pool[ci]
                cb_list = list(cb)
                
                # Remove block bi
                for idx in block_tsubs[bi]:
                    cov_count[idx] -= 1
                blocks_list.pop(bi)
                block_tsubs.pop(bi)
                n_blocks -= 1
                
                # Add candidate
                indices = block_tsub_indices(cb_list)
                blocks_list.append(cb_list)
                block_tsubs.append(indices)
                for idx in indices:
                    cov_count[idx] += 1
                n_blocks += 1
        
        # Final: ensure feasibility
        uncovered = np.where(cov_count < 1)[0]
        if len(uncovered) > 0:
            for target_idx in uncovered:
                target_ts = all_tsubs[target_idx]
                good = [cb for cb in candidate_pool if all(p in cb for p in target_ts)]
                if good:
                    cb = rng.choice(good)
                    cb_list = list(cb)
                    indices = block_tsub_indices(cb_list)
                    blocks_list.append(cb_list)
                    block_tsubs.append(indices)
                    for idx in indices:
                        cov_count[idx] += 1
                    n_blocks += 1
        
        # Use best if it's feasible and better
        cov_best = np.zeros(n_tsubs, dtype=np.int32)
        for block in best_blocks:
            for idx in block_tsub_indices(block):
                cov_best[idx] += 1
        if np.all(cov_best >= 1) and len(best_blocks) < n_blocks:
            return best_blocks, len(best_blocks)
        else:
            return blocks_list, n_blocks
    
    # Phase 1: Initial greedy coverings
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic greedy
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Randomized greedy restarts
    seed = 1
    while time.time() < deadline - 60:
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Phase 2: Deletion pass
    if time.time() < deadline - 50:
        best_blocks = deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Phase 3: WalkSAT from multiple starts
    rng = random.Random(42)
    
    # WalkSAT from best
    if time.time() < deadline - 20:
        ws_blocks, ws_count = walksat_optimize(best_blocks, rng, deadline, max_iters=150000)
        if ws_count < best_count:
            best_blocks = ws_blocks
            best_count = ws_count
    
    # WalkSAT from other starting points
    for start_seed in range(5):
        if time.time() > deadline - 10:
            break
        alt_blocks = greedy_build(200 + start_seed, randomize_order=True)
        alt_blocks = deletion_pass(alt_blocks, deadline)
        if len(alt_blocks) < best_count + 10:
            ws_blocks, ws_count = walksat_optimize(alt_blocks, rng, deadline, max_iters=100000)
            if ws_count < best_count:
                best_blocks = ws_blocks
                best_count = ws_count
    
    # Final deletion pass
    if time.time() < deadline - 3:
        best_blocks = deletion_pass(best_blocks, deadline)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)
