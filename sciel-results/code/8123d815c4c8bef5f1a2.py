"""Covering design C(v,k,t) via Simulated Annealing with multi-block moves."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 150s kill
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
        """Randomized greedy covering to get a feasible starting point."""
        rng = random.Random(seed)
        uncovered_mask = np.ones(n_tsubs, dtype=np.int32)
        blocks = []
        
        # We don't need the full tsub_order shuffle for every step if it's slow, 
        # but let's keep the logic similar to the seed for a good start.
        # Actually, for SA, a slightly worse start is fine, but a good start helps.
        
        while uncovered_mask.any():
            # Find first uncovered (in original order for determinism-ish, or random)
            # Using argmax on inverted mask is fast
            uncovered_indices = np.where(uncovered_mask == 1)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered one to break symmetry
            first_uncovered = rng.choice(uncovered_indices.tolist())
            
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
                # Check if block i is redundant
                # A block is redundant if for every t-sub it covers, 
                # there is another block covering it.
                # Equivalent: min coverage of its t-subs (excluding itself) >= 1
                # Or: for all idx in block_tsubs[i], cov_count[idx] > 1
                
                # Optimization: if any idx has cov_count == 1, block is NOT redundant.
                # If all idx have cov_count > 1, block IS redundant.
                is_redundant = True
                for idx in block_tsubs[i]:
                    if cov_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for idx in block_tsubs[i]:
                        cov_count[idx] -= 1
                    blocks_list.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    break
        
        return blocks_list

    def simulated_annealing(initial_blocks, rng, time_limit):
        """
        SA optimization.
        State: list of blocks.
        Energy: N_blocks + Lambda * N_uncovered.
        Moves: 
          1. Remove a random block.
          2. Add a random block (from pool or generated).
          3. Swap: Remove one, Add one.
        """
        blocks_list = [list(b) for b in initial_blocks]
        
        # Build coverage
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks_list:
            indices = block_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        n_blocks = len(blocks_list)
        
        # Precompute candidate pool
        # We want blocks that are "good" - cover many t-subs.
        # But for SA, we need diversity.
        # Let's generate a pool of random blocks.
        candidate_pool = []
        num_candidates = 2000
        for _ in range(num_candidates):
            cb = tuple(sorted(rng.sample(range(v), k)))
            candidate_pool.append(cb)
        
        # Also add blocks derived from uncovered t-subs dynamically? 
        # For now, static pool + on-the-fly generation if needed.
        
        best_blocks = [list(b) for b in blocks_list]
        best_energy = n_blocks # assuming feasible start
        
        # SA Parameters
        T = 100.0
        cooling_rate = 0.995
        min_T = 1.0
        
        # We need to track N_uncovered quickly.
        # n_uncovered = np.sum(cov_count == 0)
        
        # Helper to compute energy
        def get_energy(cov_count, n_blocks):
            n_unc = np.sum(cov_count == 0)
            return n_blocks + 100.0 * n_unc # Lambda = 100 to strongly penalize infeasibility
        
        current_energy = get_energy(cov_count, n_blocks)
        
        steps = 0
        last_improve_step = 0
        
        while time.time() < time_limit:
            steps += 1
            if steps % 1000 == 0:
                # print(f"SA Step {steps}, T={T:.2f}, E={current_energy:.2f}, Best={best_energy}")
                pass
            
            # Choose a move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Remove a block
                if n_blocks == 0:
                    continue
                bi = rng.randrange(n_blocks)
                # Remove
                for idx in block_tsubs[bi]:
                    cov_count[idx] -= 1
                removed_block = blocks_list.pop(bi)
                removed_tsubs = block_tsubs.pop(bi)
                n_blocks -= 1
                
                new_energy = get_energy(cov_count, n_blocks)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in blocks_list]
                        last_improve_step = steps
                else:
                    # Reject: revert
                    blocks_list.insert(bi, removed_block)
                    block_tsubs.insert(bi, removed_tsubs)
                    for idx in removed_tsubs:
                        cov_count[idx] += 1
                    n_blocks += 1
                    
            elif move_type < 0.6:
                # Add a block
                # Pick from pool or generate
                if rng.random() < 0.5 and candidate_pool:
                    cb = rng.choice(candidate_pool)
                else:
                    cb = tuple(sorted(rng.sample(range(v), k)))
                
                cb_list = list(cb)
                indices = block_tsub_indices(cb_list)
                
                # Add
                blocks_list.append(cb_list)
                block_tsubs.append(indices)
                for idx in indices:
                    cov_count[idx] += 1
                n_blocks += 1
                
                new_energy = get_energy(cov_count, n_blocks)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in blocks_list]
                        last_improve_step = steps
                else:
                    # Reject: revert
                    blocks_list.pop()
                    block_tsubs.pop()
                    for idx in indices:
                        cov_count[idx] -= 1
                    n_blocks -= 1

            else:
                # Swap: Remove one, Add one
                if n_blocks == 0:
                    continue
                bi = rng.randrange(n_blocks)
                
                # Remove
                for idx in block_tsubs[bi]:
                    cov_count[idx] -= 1
                removed_block = blocks_list.pop(bi)
                removed_tsubs = block_tsubs.pop(bi)
                n_blocks -= 1
                
                # Add
                if rng.random() < 0.5 and candidate_pool:
                    cb = rng.choice(candidate_pool)
                else:
                    cb = tuple(sorted(rng.sample(range(v), k)))
                
                cb_list = list(cb)
                indices = block_tsub_indices(cb_list)
                
                # Insert at same position to keep list consistent (though order doesn't matter for coverage)
                blocks_list.insert(bi, cb_list)
                block_tsubs.insert(bi, indices)
                for idx in indices:
                    cov_count[idx] += 1
                n_blocks += 1
                
                new_energy = get_energy(cov_count, n_blocks)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in blocks_list]
                        last_improve_step = steps
                else:
                    # Reject: revert
                    blocks_list.pop(bi)
                    block_tsubs.pop(bi)
                    for idx in indices:
                        cov_count[idx] -= 1
                    
                    blocks_list.insert(bi, removed_block)
                    block_tsubs.insert(bi, removed_tsubs)
                    for idx in removed_tsubs:
                        cov_count[idx] += 1
                
            # Cooling
            T *= cooling_rate
            if T < min_T:
                T = min_T
            
            # If stuck for a long time, restart from best with higher T?
            # Or just let it cool.
            if steps - last_improve_step > 5000:
                # Reheat
                T = 50.0
                last_improve_step = steps
                # Optionally reset to best state
                # blocks_list = [list(b) for b in best_blocks]
                # ... rebuild cov_count ...
                # This is expensive, so maybe just reheat first.
                
        return best_blocks

    # Phase 1: Initial Greedy
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic greedy
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # A few randomized greedies
    seed = 1
    while time.time() < deadline - 60:
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
        
    # Deletion pass on best
    if time.time() < deadline - 50:
        best_blocks = deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)

    # Phase 2: Simulated Annealing
    rng = random.Random(42)
    
    # Run SA from the best greedy solution
    if time.time() < deadline - 10:
        sa_blocks = simulated_annealing(best_blocks, rng, deadline - 5)
        if len(sa_blocks) < best_count:
            best_blocks = sa_blocks
            best_count = len(sa_blocks)
        
        # Run SA from a different greedy start if time permits
        if time.time() < deadline - 15:
            alt_blocks = greedy_build(123, randomize_order=True)
            alt_blocks = deletion_pass(alt_blocks, deadline)
            if len(alt_blocks) < best_count + 10: # Only if competitive
                alt_blocks = simulated_annealing(alt_blocks, rng, deadline - 5)
                if len(alt_blocks) < best_count:
                    best_blocks = alt_blocks
                    best_count = len(alt_blocks)

    # Final deletion pass
    if time.time() < deadline - 3:
        best_blocks = deletion_pass(best_blocks, deadline)

    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)
