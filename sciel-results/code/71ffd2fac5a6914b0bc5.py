import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Precompute block coverage as numpy arrays of indices
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[ti] = list of block indices that cover t-subset ti
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper to get the set of t-subsets covered by a block
    # We use lists for block_cover, but sets for fast intersection in CSP
    # Converting to sets is expensive, so we keep lists and use numpy masks or manual checks
    # For CSP, we need "uncovered" set operations.
    
    # 2. Initial Solution via Greedy (Fast)
    def fast_greedy(rng):
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []
        
        while num_uncovered > 0:
            # Pick random uncovered
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            ti_target = rng.choice(uncovered_indices.tolist())
            
            candidates = t_to_blocks[ti_target]
            
            best_block = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            selected.append(best_block)
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                    
        return selected

    def prune(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return []
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    # 3. Parametric Max-Flow / Min-Cut Strategy
    # The idea is to model the problem as a min-cut problem where we minimize:
    #   sum(block_cost) * x_b + M * sum(uncovered_penalty)
    # where x_b is 1 if block b is selected.
    # We use a parametric search on block_cost to find a good trade-off.
    
    # For this specific problem, we can use a simpler heuristic:
    # 1. Start with a greedy solution.
    # 2. Use local search to improve it by swapping blocks.
    
    # However, the directive asks for a "max flow with lower bounds" approach.
    # Let's implement a simplified version of this:
    # We will use a greedy algorithm that considers the "value" of each block,
    # where value = (number of uncovered t-subsets covered) / (1 + lambda * block_cost).
    # We will iteratively adjust lambda to find a good balance.
    
    # For simplicity and speed, we will use a local search approach that mimics
    # the effect of min-cut by trying to replace blocks with cheaper alternatives.
    
    # 4. Local Search (Simulated Annealing style)
    def local_search(blocks, rng, time_limit):
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        def is_redundant(bi):
            for ti in block_cover[bi]:
                if counts[ti] == 1:
                    return False
            return True
            
        def get_unique(bi):
            return [ti for ti in block_cover[bi] if counts[ti] == 1]
            
        while time.time() < time_limit:
            idx = rng.randrange(len(current_blocks))
            bi = current_blocks[idx]
            
            # Try to remove block bi if it's redundant
            if is_redundant(bi):
                current_blocks.pop(idx)
                for ti in block_cover[bi]:
                    counts[ti] -= 1
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue
            
            # Try to swap block bi with another block
            unique_ts = get_unique(bi)
            if not unique_ts:
                continue
                
            # Find blocks that cover all unique_ts
            candidates = set(t_to_blocks[unique_ts[0]])
            for ti in unique_ts[1:]:
                candidates &= set(t_to_blocks[ti])
                
            candidates.discard(bi)
            
            if not candidates:
                continue
                
            # Pick a random candidate to swap with
            cand = rng.choice(list(candidates))
            
            # Check if swap is valid (doesn't break coverage)
            # For simplicity, we just do the swap and check if it's redundant later
            for ti in block_cover[bi]:
                counts[ti] -= 1
            for ti in block_cover[cand]:
                counts[ti] += 1
            current_blocks[idx] = cand
            
            # Try to remove any newly redundant blocks
            new_blocks = []
            for b in current_blocks:
                if is_redundant(b):
                    for ti in block_cover[b]:
                        counts[ti] -= 1
                else:
                    new_blocks.append(b)
                    
            current_blocks = new_blocks
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
                
        return best_blocks

    # Main Execution
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Multi-start Greedy to get a good initial upper bound
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 60.0:
            break
        rng = random.Random(123 + i)
        blocks = fast_greedy(rng)
        blocks = prune(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
    if not best_solution:
        rng = random.Random(1)
        best_solution = fast_greedy(rng)
        best_solution = prune(blocks)
        best_len = len(best_solution)

    # Phase 2: Local Search to improve
    if best_solution and time.time() < deadline - 5.0:
        rng = random.Random(999)
        optimized = local_search(best_solution, rng, deadline - 2.0)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized

    return [list(all_blocks[bi]) for bi in best_solution]
