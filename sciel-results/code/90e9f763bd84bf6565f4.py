"""Covering design C(v,k,t): bitset-accelerated randomized greedy with redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math
from collections import Counter


def _all_t_subsets(v, t):
    """Return list of all t-subsets as sorted tuples."""
    return list(itertools.combinations(range(v), t))


def _subset_to_idx(t_subs):
    """Map each t-subset tuple to its index."""
    return {s: i for i, s in enumerate(t_subs)}


def _block_tsub_indices(block, t, idx_map):
    """Return set of indices of t-subsets covered by block."""
    return {idx_map[comb] for comb in itertools.combinations(block, t)}


def _greedy_pass(v, k, t, rng=None):
    """One greedy pass using bitmasks for speed. Returns list of blocks."""
    if rng is None:
        rng = random.Random()
    
    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx(t_subs)
    n_tsubs = len(t_subs)
    
    # Use a bitarray-like structure: list of booleans for uncovered status
    uncovered = [True] * n_tsubs
    uncovered_count = n_tsubs
    blocks = []
    
    # Precompute for each point x, the list of t-subsets that contain x
    # and for each (x, block_so_far), we need to compute gain efficiently
    # For now, use a simpler approach with caching
    
    while uncovered_count > 0:
        # Pick a random uncovered t-subset to start the block
        # Find an uncovered index
        while True:
            idx = rng.randrange(n_tsubs)
            if uncovered[idx]:
                break
        start_sub = t_subs[idx]
        block = list(start_sub)
        
        # Add points greedily
        while len(block) < k:
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]
            best_x = None
            best_gain = -1
            
            # Precompute t-1 subsets of current block
            t1_subs = list(itertools.combinations(block, t - 1))
            
            for x in candidates:
                gain = 0
                for rest in t1_subs:
                    # rest is a tuple of t-1 elements, add x
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in idx_map:
                        i = idx_map[new_sub]
                        if uncovered[i]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for comb in itertools.combinations(block, t):
            i = idx_map[comb]
            if uncovered[i]:
                uncovered[i] = False
                uncovered_count -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, rng=None):
    """Try to remove redundant blocks from a valid covering using bitmasks."""
    if not blocks:
        return blocks
    if rng is None:
        rng = random.Random(0)
    
    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx(t_subs)
    n_tsubs = len(t_subs)
    
    # For each block, compute the set of t-subset indices it covers
    block_masks = []
    for b in blocks:
        mask = set()
        for comb in itertools.combinations(b, t):
            mask.add(idx_map[comb])
        block_masks.append(mask)
    
    n_blocks = len(blocks)
    
    # Count how many blocks cover each t-subset
    coverage_count = [0] * n_tsubs
    for mask in block_masks:
        for i in mask:
            coverage_count[i] += 1
    
    # Try to remove blocks in random order (multiple passes for better results)
    kept_indices = list(range(n_blocks))
    changed = True
    max_passes = 5
    
    for _ in range(max_passes):
        if not changed:
            break
        changed = False
        rng.shuffle(kept_indices)
        
        new_kept = []
        for i in kept_indices:
            # Check if block i is redundant: all its t-subsets are covered by other kept blocks
            redundant = True
            for tsub_idx in block_masks[i]:
                if coverage_count[tsub_idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_kept.append(i)
            else:
                # Remove block i
                for tsub_idx in block_masks[i]:
                    coverage_count[tsub_idx] -= 1
                changed = True
        
        kept_indices = new_kept
    
    return [blocks[i] for i in kept_indices]


def _simulated_annealing(v, k, t, initial_blocks, candidate_pool, rng, deadline):
    """
    Perform simulated annealing to minimize the number of blocks in a valid covering.
    State: set of block indices from candidate_pool.
    Objective: Minimize (num_uncovered * 10000 + len(blocks)).
    """
    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx(t_subs)
    n_tsubs = len(t_subs)
    
    # Precompute t-subset indices for all candidate blocks
    # candidate_pool is a list of blocks (lists of points)
    # Map each candidate block to its t-subset indices
    # To save memory/time, we only keep indices for blocks in the pool
    block_indices_list = []
    for b in candidate_pool:
        indices = set()
        for comb in itertools.combinations(b, t):
            indices.add(idx_map[comb])
        block_indices_list.append(indices)
        
    # Initial state: indices of initial_blocks in candidate_pool
    # We need to map initial_blocks to their indices in candidate_pool
    # If initial_blocks are not in candidate_pool, add them? 
    # Better: construct candidate_pool to include initial_blocks first.
    
    # Map block tuple to index in pool for O(1) lookup
    block_to_idx = {}
    for i, b in enumerate(candidate_pool):
        block_to_idx[tuple(sorted(b))] = i
        
    current_indices = set()
    for b in initial_blocks:
        idx = block_to_idx[tuple(sorted(b))]
        current_indices.add(idx)
        
    # Calculate initial coverage
    coverage_count = [0] * n_tsubs
    for idx in current_indices:
        for tsub_idx in block_indices_list[idx]:
            coverage_count[tsub_idx] += 1
            
    num_uncovered = 0
    for c in coverage_count:
        if c == 0:
            num_uncovered += 1
            
    current_cost = num_uncovered * 10000 + len(current_indices)
    best_indices = set(current_indices)
    best_cost = current_cost
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    steps = 0
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random block from pool
            if not current_indices:
                continue
            # Pick a random block not in current set
            # Optimization: pick random index, if in set, retry or pick different
            # For simplicity, pick random index from pool
            candidate_idx = rng.randrange(len(candidate_pool))
            if candidate_idx in current_indices:
                continue # Skip if already present
            
            # Calculate delta cost
            # Adding block i increases coverage of its tsubs
            delta_uncovered = 0
            for tsub_idx in block_indices_list[candidate_idx]:
                if coverage_count[tsub_idx] == 0:
                    delta_uncovered -= 1 # Covered one more
                coverage_count[tsub_idx] += 1
            
            # Tentative apply
            current_indices.add(candidate_idx)
            new_cost = (num_uncovered + delta_uncovered) * 10000 + len(current_indices)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_indices = set(current_indices)
            else:
                # Accept with probability exp(-delta/T)
                delta = new_cost - current_cost
                if delta > 0:
                    if rng.random() < math.exp(-delta / T):
                        current_cost = new_cost
                        num_uncovered += delta_uncovered
                    else:
                        # Reject: revert
                        current_indices.remove(candidate_idx)
                        for tsub_idx in block_indices_list[candidate_idx]:
                            coverage_count[tsub_idx] -= 1
                else:
                    # Accept (improvement)
                    current_cost = new_cost
                    num_uncovered += delta_uncovered
                    
        elif move_type < 0.8:
            # Remove a random block from current set
            if not current_indices:
                continue
            candidate_idx = rng.choice(list(current_indices))
            
            # Calculate delta cost
            delta_uncovered = 0
            for tsub_idx in block_indices_list[candidate_idx]:
                if coverage_count[tsub_idx] == 1:
                    delta_uncovered += 1 # Becomes uncovered
                coverage_count[tsub_idx] -= 1
                
            # Tentative apply
            current_indices.remove(candidate_idx)
            new_cost = (num_uncovered + delta_uncovered) * 10000 + len(current_indices)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_indices = set(current_indices)
            else:
                delta = new_cost - current_cost
                if delta > 0:
                    if rng.random() < math.exp(-delta / T):
                        current_cost = new_cost
                        num_uncovered += delta_uncovered
                    else:
                        # Reject: revert
                        current_indices.add(candidate_idx)
                        for tsub_idx in block_indices_list[candidate_idx]:
                            coverage_count[tsub_idx] += 1
                else:
                    current_cost = new_cost
                    num_uncovered += delta_uncovered
                    
        else:
            # Swap: Remove one, Add one
            if not current_indices:
                continue
            remove_idx = rng.choice(list(current_indices))
            add_idx = rng.randrange(len(candidate_pool))
            if add_idx == remove_idx or add_idx in current_indices:
                continue
                
            # Calculate delta for removal
            delta_rem_uncovered = 0
            for tsub_idx in block_indices_list[remove_idx]:
                if coverage_count[tsub_idx] == 1:
                    delta_rem_uncovered += 1
                coverage_count[tsub_idx] -= 1
                
            # Calculate delta for addition
            delta_add_uncovered = 0
            for tsub_idx in block_indices_list[add_idx]:
                if coverage_count[tsub_idx] == 0:
                    delta_add_uncovered -= 1
                coverage_count[tsub_idx] += 1
                
            total_delta_uncovered = delta_rem_uncovered + delta_add_uncovered
            delta_len = 0 # Swap keeps size same
            
            # Tentative apply
            current_indices.remove(remove_idx)
            current_indices.add(add_idx)
            
            new_cost = (num_uncovered + total_delta_uncovered) * 10000 + len(current_indices)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_indices = set(current_indices)
            else:
                delta = new_cost - current_cost
                if delta > 0:
                    if rng.random() < math.exp(-delta / T):
                        current_cost = new_cost
                        num_uncovered += total_delta_uncovered
                    else:
                        # Reject: revert
                        current_indices.remove(add_idx)
                        current_indices.add(remove_idx)
                        # Revert coverage counts
                        for tsub_idx in block_indices_list[add_idx]:
                            coverage_count[tsub_idx] -= 1
                        for tsub_idx in block_indices_list[remove_idx]:
                            coverage_count[tsub_idx] += 1
                else:
                    current_cost = new_cost
                    num_uncovered += total_delta_uncovered

        T *= alpha
        steps += 1
        
    # Return the best set of blocks found
    return [candidate_pool[i] for i in best_indices]


def construct(v, k, t):
    """Find a covering design using randomized greedy with redundancy removal and simulated annealing."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 115s out of 150s limit
    
    best_blocks = None
    best_len = float('inf')
    
    # 1. Generate initial covering using greedy
    rng = random.Random(42)
    greedy_blocks = _greedy_pass(v, k, t, rng)
    greedy_blocks = _remove_redundant_blocks(greedy_blocks, v, k, t, rng)
    
    if len(greedy_blocks) < best_len:
        best_len = len(greedy_blocks)
        best_blocks = greedy_blocks
        
    # 2. Build candidate pool
    # Include greedy blocks and some random blocks
    candidate_pool = list(greedy_blocks)
    
    # Add random blocks to pool to allow SA to explore
    # Number of random blocks: proportional to v, k
    num_random_blocks = 200 
    for _ in range(num_random_blocks):
        block = rng.sample(range(v), k)
        block.sort()
        # Avoid duplicates in pool
        if block not in candidate_pool:
            candidate_pool.append(block)
            
    # 3. Run Simulated Annealing
    # We want to finish before deadline.
    # SA is stochastic, so we run it until time runs out or T is too low.
    # We use the greedy result as the initial state.
    # Note: SA might degrade the solution if not careful, but we track best.
    
    sa_deadline = time.time() + 60 # Use 60s for SA
    if time.time() < sa_deadline:
        sa_blocks = _simulated_annealing(v, k, t, greedy_blocks, candidate_pool, rng, sa_deadline)
        
        # Verify validity of SA result (SA objective should ensure it, but double check)
        # And remove redundancies
        if sa_blocks:
            # Quick validity check
            t_subs = _all_t_subsets(v, t)
            idx_map = _subset_to_idx(t_subs)
            covered = set()
            for b in sa_blocks:
                for comb in itertools.combinations(b, t):
                    covered.add(idx_map[comb])
                    
            if len(covered) == len(t_subs):
                sa_blocks = _remove_redundant_blocks(sa_blocks, v, k, t, rng)
                if len(sa_blocks) < best_len:
                    best_len = len(sa_blocks)
                    best_blocks = sa_blocks
    
    # 4. If time remains, try a few more greedy restarts with different seeds
    while time.time() < deadline - 5:
        attempt = int(time.time()) # Change seed based on time
        rng2 = random.Random(attempt)
        blocks = _greedy_pass(v, k, t, rng2)
        blocks = _remove_redundant_blocks(blocks, v, k, t, rng2)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback
        best_blocks = _greedy_pass(v, k, t)
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t)
        
    return best_blocks
# EVOLVE-BLOCK-END
