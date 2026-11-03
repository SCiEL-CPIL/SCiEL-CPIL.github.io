"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, point_to_subsets):
    """Fast greedy construction to get a valid starting covering."""
    n_subsets = len(all_subsets)
    uncovered = [True] * n_subsets
    uncovered_count = n_subsets
    blocks = []
    block_set = set()
    
    while uncovered_count > 0:
        # Pick a random uncovered t-subset as seed
        while True:
            seed_idx = rng.randrange(n_subsets)
            if uncovered[seed_idx]:
                break
        seed = list(all_subsets[seed_idx])
        rng.shuffle(seed)
        block = list(seed)
        block_set_local = set(block)
        
        # Greedily add points to maximize coverage of uncovered subsets
        while len(block) < k:
            best_x = None
            best_gain = 0
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_idx.get(fs)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain == 0:
                break
            block.append(best_x)
            block_set_local.add(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
    
    return blocks


def _block_covered_indices(block, subset_to_idx):
    """Return the list of t-subset indices covered by a block."""
    covered = []
    for c in itertools.combinations(block, t):
        idx = subset_to_idx[frozenset(c)]
        covered.append(idx)
    return covered


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 112.0  # leave 8s buffer for safety

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    
    # Precompute for each point, the list of t-subsets containing it
    point_to_subsets = [[] for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].append(idx)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, point_to_subsets)
    
    # Precompute covered subsets for each block
    block_covered = [_block_covered_indices(b, subset_to_idx) for b in blocks]
    
    # counts[i] = number of blocks covering subset i
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Simulated Annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    iteration = 0
    no_improve = 0
    max_no_improve = 500
    
    def try_remove_redundant():
        """Try to remove a redundant block. Returns True if removed."""
        if not blocks:
            return False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                nonlocal best_count, best_blocks, no_improve
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                return True
        return False
    
    def try_swap_block():
        """Try to swap a block with a better one. Returns True if swapped."""
        if not blocks:
            return False
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        old_covered = block_covered[i]
        old_block_set = set(old_block)
        
        # Identify critical subsets (count == 1) in old_block
        critical = [idx for idx in old_covered if counts[idx] == 1]
        
        # Collect points needed for critical subsets
        points_needed = set()
        for idx in critical:
            points_needed.update(all_subsets[idx])
        
        # If we can't fit all critical points in one block, skip
        if len(points_needed) > k:
            return False
        
        # Build new block containing points_needed
        new_block_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(cands)
        
        # Greedy fill to maximize coverage of low-count subsets
        for x in cands:
            if len(new_block_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_block_set, t - 1):
                fs = frozenset(list(rest) + [x])
                if fs in subset_to_idx:
                    idx = subset_to_idx[fs]
                    if counts[idx] == 0:
                        gain += 10
                    elif counts[idx] == 1:
                        gain += 1
            if gain > 0:
                new_block_set.add(x)
            elif len(new_block_set) < k:
                new_block_set.add(x)
        
        if len(new_block_set) != k:
            return False
        
        new_block = tuple(sorted(new_block_set))
        
        if new_block == old_block:
            return False
        
        new_covered = _block_covered_indices(new_block, subset_to_idx)
        new_covered_set = set(new_covered)
        
        # Check if swap is valid: no subset loses its last cover
        valid = True
        for idx in old_covered:
            if idx not in new_covered_set and counts[idx] <= 1:
                valid = False
                break
        
        if valid:
            for idx in old_covered:
                counts[idx] -= 1
            for idx in new_covered:
                counts[idx] += 1
            
            blocks[i] = new_block
            block_covered[i] = new_covered
            no_improve = 0
            return True
        return False
    
    def try_random_swap():
        """Try a random 1-in-1-out swap. Returns True if swapped."""
        if not blocks:
            return False
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        old_covered = block_covered[i]
        old_block_set = set(old_block)
        
        # Remove one random element
        rem = rng.choice(list(old_block_set))
        new_block_set = set(old_block_set)
        new_block_set.remove(rem)
        
        # Add a random new element
        cands = [x for x in range(v) if x not in new_block_set]
        if not cands:
            return False
        new_block_set.add(rng.choice(cands))
        
        new_block = tuple(sorted(new_block_set))
        if new_block == old_block:
            return False
        
        new_covered = _block_covered_indices(new_block, subset_to_idx)
        new_covered_set = set(new_covered)
        
        # Check if swap is valid
        valid = True
        for idx in old_covered:
            if idx not in new_covered_set and counts[idx] <= 1:
                valid = False
                break
        
        if valid:
            for idx in old_covered:
                counts[idx] -= 1
            for idx in new_covered:
                counts[idx] += 1
            
            blocks[i] = new_block
            block_covered[i] = new_covered
            no_improve = 0
            return True
        return False
    
    def restart_greedy():
        """Restart with a fresh greedy cover if it's better."""
        new_cover = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, point_to_subsets)
        if len(new_cover) < len(blocks):
            blocks.clear()
            block_covered.clear()
            for b in new_cover:
                blocks.append(b)
                block_covered.append(_block_covered_indices(b, subset_to_idx))
            counts[:] = [0] * n_subsets
            for covered in block_covered:
                for idx in covered:
                    counts[idx] += 1
            nonlocal best_count, best_blocks
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
            no_improve = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block
        if try_remove_redundant():
            continue
        
        # 2. Try to swap a block (targeted)
        if try_swap_block():
            continue
        
        # 3. Try random swap
        if try_random_swap():
            continue
        
        # 4. Restart if stuck for too long
        no_improve += 1
        if no_improve > max_no_improve:
            restart_greedy()
            no_improve = 0
        
        # Cool down
        T = max(T_min, T * cooling)
    
    return best_blocks
# EVOLVE-BLOCK-END
