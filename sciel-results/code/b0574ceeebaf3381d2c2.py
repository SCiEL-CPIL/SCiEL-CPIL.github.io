"""Covering design C(v,k,t) with simulated annealing + pair-exchange moves."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

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
                # Check if removing block i keeps all covered
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
    
    def simulated_annealing(initial_blocks, rng, deadline, max_iters=500000):
        """
        Simulated annealing over block set.
        State: list of blocks.
        Energy: n_blocks + lambda * (number of uncovered t-subs)
        Moves: 
          - Remove a block (if feasible, i.e., no t-sub becomes uncovered)
          - Add a block (from candidate pool)
          - Pair exchange: remove one, add another
        Temperature controls acceptance of infeasible/worse states.
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
        
        def get_energy():
            """Energy = n_blocks + penalty for uncovered."""
            uncovered = np.sum(cov_count < 1)
            return n_blocks + 10.0 * uncovered
        
        # Candidate pool for additions
        candidate_pool = set()
        for _ in range(min(3000, n_tsubs)):
            cb = tuple(sorted(rng.sample(range(v), k)))
            candidate_pool.add(cb)
        # Add blocks extending random t-subs
        for _ in range(1000):
            ts = rng.choice(all_tsubs)
            remaining = [x for x in range(v) if x not in ts]
            extra = rng.sample(remaining, k - t)
            cb = tuple(sorted(list(ts) + extra))
            candidate_pool.add(cb)
        candidate_pool = list(candidate_pool)
        
        # Temperature schedule
        T0 = 5.0
        T_end = 0.01
        cooling = (T_end / T0) ** (1.0 / max(max_iters, 1))
        T = T0
        
        best_blocks = [list(b) for b in blocks_list]
        best_energy = get_energy()
        best_n_blocks = n_blocks
        
        no_improve = 0
        max_no_improve = 20000
        
        iter_count = 0
        while iter_count < max_iters and time.time() < deadline - 5:
            iter_count += 1
            T *= cooling
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.3 and n_blocks > 1:
                # Try to remove a block
                bi = rng.randrange(n_blocks)
                critical = 0
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        critical += 1
                
                if critical == 0:
                    # Safe removal
                    for idx in block_tsubs[bi]:
                        cov_count[idx] -= 1
                    blocks_list.pop(bi)
                    block_tsubs.pop(bi)
                    n_blocks -= 1
                    
                    energy = get_energy()
                    if energy < best_energy:
                        best_energy = energy
                        best_n_blocks = n_blocks
                        best_blocks = [list(b) for b in blocks_list]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Infeasible removal - accept with probability exp(-delta/T)
                    # delta = increase in penalty
                    delta = 10.0 * critical - 1.0  # lose 1 block, gain penalty
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        for idx in block_tsubs[bi]:
                            cov_count[idx] -= 1
                        blocks_list.pop(bi)
                        block_tsubs.pop(bi)
                        n_blocks -= 1
                        no_improve += 1
                    else:
                        no_improve += 1
            
            elif move_type < 0.6:
                # Add a random block
                if candidate_pool:
                    cb = rng.choice(candidate_pool)
                    cb_list = list(cb)
                    indices = block_tsub_indices(cb_list)
                    
                    # Check if it helps (covers some uncovered)
                    helps = 0
                    for idx in indices:
                        if cov_count[idx] == 0:
                            helps += 1
                    
                    if helps > 0:
                        blocks_list.append(cb_list)
                        block_tsubs.append(indices)
                        for idx in indices:
                            cov_count[idx] += 1
                        n_blocks += 1
                        no_improve += 1
                    else:
                        no_improve += 1
            
            else:
                # Pair exchange: remove one, add another
                if n_blocks > 1 and candidate_pool:
                    bi = rng.randrange(n_blocks)
                    # Remove block bi
                    for idx in block_tsubs[bi]:
                        cov_count[idx] -= 1
                    blocks_list.pop(bi)
                    block_tsubs.pop(bi)
                    n_blocks -= 1
                    
                    # Add a candidate
                    cb = rng.choice(candidate_pool)
                    cb_list = list(cb)
                    indices = block_tsub_indices(cb_list)
                    blocks_list.append(cb_list)
                    block_tsubs.append(indices)
                    for idx in indices:
                        cov_count[idx] += 1
                    n_blocks += 1
                    
                    energy = get_energy()
                    if energy < best_energy:
                        best_energy = energy
                        best_n_blocks = n_blocks
                        best_blocks = [list(b) for b in blocks_list]
                        no_improve = 0
                    else:
                        no_improve += 1
            
            # Periodic feasibility repair
            if no_improve % 5000 == 0:
                uncovered = np.where(cov_count < 1)[0]
                if len(uncovered) > 0:
                    for target_idx in uncovered[:10]:
                        target_ts = all_tsubs[target_idx]
                        # Find candidate covering this t-sub
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
        # Check best_blocks feasibility
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
    while time.time() < deadline - 50:
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Phase 2: Deletion pass
    if time.time() < deadline - 40:
        best_blocks = deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Phase 3: Simulated annealing from multiple starts
    rng = random.Random(42)
    
    # SA from best
    if time.time() < deadline - 20:
        sa_blocks, sa_count = simulated_annealing(best_blocks, rng, deadline, max_iters=300000)
        if sa_count < best_count:
            best_blocks = sa_blocks
            best_count = sa_count
    
    # SA from other starting points
    for start_seed in range(5):
        if time.time() > deadline - 10:
            break
        alt_blocks = greedy_build(200 + start_seed, randomize_order=True)
        alt_blocks = deletion_pass(alt_blocks, deadline)
        if len(alt_blocks) < best_count + 10:
            sa_blocks, sa_count = simulated_annealing(alt_blocks, rng, deadline, max_iters=200000)
            if sa_count < best_count:
                best_blocks = sa_blocks
                best_count = sa_count
    
    # Final deletion pass
    if time.time() < deadline - 3:
        best_blocks = deletion_pass(best_blocks, deadline)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)
