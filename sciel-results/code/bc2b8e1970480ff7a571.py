"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
import numpy as np


def _precompute_triples(v, t):
    """Precompute all t-subsets as frozensets and their indices."""
    triples = [frozenset(c) for c in itertools.combinations(range(v), t)]
    triple_index = {f: i for i, f in enumerate(triples)}
    return triples, triple_index


def _block_triple_indices(block, t, triple_index):
    """Get indices of all t-subsets in a block."""
    return [triple_index[frozenset(c)] for c in itertools.combinations(block, t)]


def _greedy_cover_bitmask(v, k, t, rng, triples, triple_index):
    """Fast greedy construction using bitmask-based coverage tracking."""
    n_triples = len(triples)
    # Use numpy boolean array for uncovered tracking
    uncovered = np.ones(n_triples, dtype=np.bool_)
    blocks = []
    blocks_set = set()
    
    while uncovered.any():
        # Pick a random uncovered t-subset as seed
        unc_indices = np.where(uncovered)[0]
        seed_idx = unc_indices[rng.randrange(len(unc_indices))]
        seed = set(triples[seed_idx])
        block_set = set(seed)
        
        # Greedily add points to maximize coverage
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = triple_index.get(fs)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block_set.add(best_x)
            
        block = tuple(sorted(block_set))
        if block not in blocks_set:
            blocks.append(block)
            blocks_set.add(block)
            for idx in _block_triple_indices(block, t, triple_index):
                uncovered[idx] = False
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute triple structure
    triples, triple_index = _precompute_triples(v, t)
    n_triples = len(triples)
    
    # Build initial valid covering
    blocks = _greedy_cover_bitmask(v, k, t, rng, triples, triple_index)
    
    # Use numpy array for coverage counts
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for idx in _block_triple_indices(b, t, triple_index):
            counts[idx] += 1
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    uncovered = int(np.sum(counts == 0))
    
    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        nonlocal uncovered
        delta_uncovered = 0
        for idx in _block_triple_indices(blk, t, triple_index):
            old_c = counts[idx]
            new_c = old_c + sign
            if new_c == 0:
                if old_c == 1:
                    delta_uncovered += 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
            counts[idx] = new_c
        uncovered += delta_uncovered
        return delta_uncovered

    def try_remove_redundant():
        """Try to remove redundant blocks. Returns True if any removed."""
        nonlocal best_blocks, best_count
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for idx in _block_triple_indices(b, t, triple_index):
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                if uncovered == 0:
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    break
                else:
                    apply_block_change(b, 1)
        return removed_any

    def try_swap_block(i):
        """Try to replace block i with a better one. Returns True if improved."""
        nonlocal best_blocks, best_count
        old_block = blocks[i]
        
        # Find critical t-subsets (covered only by this block)
        critical_indices = []
        for idx in _block_triple_indices(old_block, t, triple_index):
            if counts[idx] == 1:
                critical_indices.append(idx)
        
        if not critical_indices:
            # Can remove this block
            delta_u = apply_block_change(old_block, -1)
            if uncovered == 0:
                blocks.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                return True
            else:
                apply_block_change(old_block, 1)
                return False
        
        # Need to find a new block that covers all critical t-subsets
        points_needed = set()
        for idx in critical_indices:
            points_needed.update(triples[idx])
        
        if len(points_needed) > k:
            return False
        
        # Generate candidate new blocks
        new_block_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(cands)
        
        # Try to fill the rest with points that maximize coverage
        for x in cands:
            if len(new_block_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_block_set, t-1):
                fs = frozenset(list(rest) + [x])
                idx = triple_index.get(fs)
                if idx is not None:
                    c = counts[idx]
                    if c == 0:
                        gain += 10
                    elif c == 1:
                        gain += 1
            new_block_set.add(x)
            
        if len(new_block_set) == k:
            new_block = tuple(sorted(new_block_set))
            if new_block != old_block:
                delta_u_remove = apply_block_change(old_block, -1)
                delta_u_add = apply_block_change(new_block, 1)
                
                if uncovered == 0:
                    blocks[i] = new_block
                    return True
                else:
                    apply_block_change(new_block, -1)
                    apply_block_change(old_block, 1)
        return False

    def perturb():
        """Perturbation: remove a block and re-add to escape local minima."""
        nonlocal best_blocks, best_count
        
        old_blocks = list(blocks)
        old_counts = counts.copy()
        old_uncovered = uncovered
        
        i = rng.randrange(len(blocks))
        b = blocks[i]
        
        delta_u = apply_block_change(b, -1)
        blocks.pop(i)
        
        # Re-add blocks to cover uncovered t-subsets
        unc_indices = np.where(counts == 0)[0]
        
        while len(unc_indices) > 0:
            seed_idx = unc_indices[rng.randrange(len(unc_indices))]
            seed = set(triples[seed_idx])
            new_b = list(seed)
            rng.shuffle(new_b)
            
            while len(new_b) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in new_b]
                rng.shuffle(cands)
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(new_b, t-1):
                        fs = frozenset(list(rest) + [x])
                        idx = triple_index.get(fs)
                        if idx is not None and counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_b.append(best_x)
            
            new_b.sort()
            new_b_tuple = tuple(new_b)
            
            if new_b_tuple not in blocks:
                blocks.append(new_b_tuple)
                delta_u_add = apply_block_change(new_b_tuple, 1)
                unc_indices = np.where(counts == 0)[0]
        
        if uncovered == 0:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
            return True
        else:
            blocks = old_blocks
            counts[:] = old_counts
            uncovered = old_uncovered
            return False

    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.9998
    min_temp = 0.001
    
    temp = initial_temp
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit and temp > min_temp:
        
        # 1. Try to remove redundant blocks
        if try_remove_redundant():
            no_improve = 0
            continue
            
        # 2. Try swaps
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            if try_swap_block(i):
                no_improve = 0
                continue
        
        # 3. Perturbation with simulated annealing acceptance
        if no_improve > max_no_improve:
            success = perturb()
            if success:
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        temp *= cooling_rate
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
