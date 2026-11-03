import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _precompute(v, k, t, time_budget):
    """Precompute all t-subsets, all k-blocks, and incidence structures."""
    start = time.time()
    
    # Generate t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Build incidence: for each k-block, which t-subsets does it cover
    block_t_sets = []
    for block in k_blocks:
        covered = frozenset(t_index[c] for c in itertools.combinations(block, t))
        block_t_sets.append(covered)
    
    # Build reverse incidence: for each t-subset, which k-blocks cover it
    t_to_blocks = defaultdict(list)
    for i, covered in enumerate(block_t_sets):
        for idx in covered:
            t_to_blocks[idx].append(i)
    t_to_blocks = [tuple(t_to_blocks[i]) for i in range(num_t)]
    
    elapsed = time.time() - start
    remaining = max(0, time_budget - elapsed)
    
    return {
        't_subsets': t_subsets,
        'k_blocks': k_blocks,
        'block_t_sets': block_t_sets,
        't_to_blocks': t_to_blocks,
        'num_t': num_t,
        'num_k': num_k,
        'remaining': remaining,
        'elapsed': elapsed
    }


def _greedy_cover(data, weights, deadline, rng):
    """Run weighted greedy set cover with optional randomization."""
    num_t = data['num_t']
    num_k = data['num_k']
    block_t_sets = data['block_t_sets']
    t_to_blocks = data['t_to_blocks']
    
    # Initialize gain for each block
    gain = np.zeros(num_k, dtype=np.float64)
    for i in range(num_k):
        if block_t_sets[i]:
            gain[i] = np.sum(weights[list(block_t_sets[i])])
    
    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)
    
    while uncovered_count > 0 and time.time() < deadline:
        # Add small random perturbation for diversity
        if rng is not None:
            perturbation = rng.uniform(0, 0.1) * (1.0 / (num_k + 1))
            gain_no_mask = gain + perturbation
        else:
            gain_no_mask = gain
        
        gain_with_mask = gain_no_mask.copy()
        gain_with_mask[selected_mask] = -np.inf
        
        best_idx = int(np.argmax(gain_with_mask))
        best_gain = gain_with_mask[best_idx]
        
        if np.isinf(best_gain) or best_gain <= 0:
            break
        
        selected.append(best_idx)
        selected_mask[best_idx] = True
        
        # Update uncovered and gains
        for idx in block_t_sets[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                w = weights[idx]
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w
    
    return selected


def _remove_redundant(selected, block_t_sets, num_t, deadline):
    """Remove redundant blocks from the selected set."""
    if not selected or time.time() > deadline:
        return selected
    
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in selected:
        for idx in block_t_sets[i]:
            cover_count[idx] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for i in selected:
            needed = False
            for idx in block_t_sets[i]:
                if cover_count[idx] == 1:
                    needed = True
                    break
            if needed:
                new_selected.append(i)
            else:
                changed = True
                for idx in block_t_sets[i]:
                    cover_count[idx] -= 1
        if changed:
            selected = new_selected
    
    return selected


def _local_improve(selected, block_t_sets, t_to_blocks, num_t, num_k, deadline, rng):
    """Try to improve by swapping blocks to reduce redundancy."""
    if not selected or time.time() > deadline:
        return selected
    
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in selected:
        for idx in block_t_sets[i]:
            cover_count[idx] += 1
    
    selected_set = set(selected)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        current_size = len(selected)
        
        # Try to remove any block that's not needed
        for i in list(selected):
            if cover_count is None:
                break
            needed = False
            for idx in block_t_sets[i]:
                if cover_count[idx] == 1:
                    needed = True
                    break
            if not needed:
                for idx in block_t_sets[i]:
                    cover_count[idx] -= 1
                selected.remove(i)
                selected_set.discard(i)
                improved = True
        
        # Try to swap: remove a block and add a better one
        if len(selected) > 0 and time.time() < deadline:
            # Find blocks with high redundancy
            redundancy = np.zeros(num_k, dtype=np.int32)
            for i in selected:
                for idx in block_t_sets[i]:
                    if cover_count[idx] > 1:
                        redundancy[i] += 1
            
            # Sort by redundancy (try removing high-redundancy blocks first)
            candidates = sorted(selected, key=lambda x: -redundancy[x])[:10]
            
            for rem_idx in candidates:
                if time.time() > deadline:
                    break
                if rem_idx not in selected:
                    continue
                
                # Temporarily remove
                for idx in block_t_sets[rem_idx]:
                    cover_count[idx] -= 1
                
                # Check if coverage is still complete
                if np.all(cover_count > 0):
                    # Try to find a block that covers more uncovered or reduces redundancy
                    best_gain = -1
                    best_add = -1
                    for b in range(num_k):
                        if b in selected_set:
                            continue
                        gain = 0
                        for idx in block_t_sets[b]:
                            if cover_count[idx] == 0:
                                gain += 10  # Priority for uncovered
                            elif cover_count[idx] == 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_add = b
                    
                    if best_add >= 0 and best_gain > 0:
                        # Perform swap
                        selected.remove(rem_idx)
                        selected_set.discard(rem_idx)
                        selected.append(best_add)
                        selected_set.add(best_add)
                        for idx in block_t_sets[best_add]:
                            cover_count[idx] += 1
                        improved = True
                else:
                    # Restore
                    for idx in block_t_sets[rem_idx]:
                        cover_count[idx] += 1
    
    return _remove_redundant(selected, block_t_sets, num_t, deadline)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute
    data = _precompute(v, k, t, 20)
    num_t = data['num_t']
    num_k = data['num_k']
    block_t_sets = data['block_t_sets']
    t_to_blocks = data['t_to_blocks']
    k_blocks = data['k_blocks']
    
    if elapsed() > deadline - 30:
        # Emergency fallback: simple greedy
        weights = np.ones(num_t, dtype=np.float64)
        selected = _greedy_cover(data, weights, deadline - 5, None)
        selected = _remove_redundant(selected, block_t_sets, num_t, deadline - 2)
        return [list(k_blocks[i]) for i in selected]
    
    # Compute rarity-based weights
    t_cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        for idx in block_t_sets[i]:
            t_cover_count[idx] += 1
    
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    base_weights = 1.0 / safe_counts
    
    best_selected = None
    best_count = float('inf')
    
    # Multiple restarts with different weight strategies
    max_restarts = 8
    restart_time_budget = (deadline - 40) / max_restarts
    
    for restart in range(max_restarts):
        if elapsed() > deadline - 35:
            break
        
        # Create diverse weights
        if restart == 0:
            weights = base_weights.copy()
        elif restart == 1:
            # Boost rare t-subsets more
            weights = base_weights ** 2
        elif restart == 2:
            # Uniform weights
            weights = np.ones(num_t, dtype=np.float64)
        else:
            # Random perturbation of base weights
            rng = random.Random(restart * 12345 + 67890)
            perturbation = np.array([rng.uniform(0.5, 1.5) for _ in range(num_t)])
            weights = base_weights * perturbation
        
        # Run greedy with optional randomization
        rng = random.Random(restart * 98765 + 43210) if restart > 0 else None
        selected = _greedy_cover(data, weights, deadline - 25, rng)
        
        # Remove redundant blocks
        selected = _remove_redundant(selected, block_t_sets, num_t, deadline - 15)
        
        # Local improvement
        if elapsed() < deadline - 10 and len(selected) < best_count:
            selected = _local_improve(
                selected, block_t_sets, t_to_blocks, num_t, num_k,
                deadline - 5, random.Random(restart * 54321 + 10987)
            )
        
        # Check coverage
        if selected:
            cover_check = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_sets[i]:
                    cover_check[idx] += 1
            
            if np.all(cover_check > 0) and len(selected) < best_count:
                best_count = len(selected)
                best_selected = selected[:]
        
        if elapsed() > deadline - 30:
            break
    
    # Final fallback if no valid solution found
    if best_selected is None:
        weights = base_weights.copy()
        selected = _greedy_cover(data, weights, deadline - 5, None)
        selected = _remove_redundant(selected, block_t_sets, num_t, deadline - 2)
        best_selected = selected
    
    return [list(k_blocks[i]) for i in best_selected]
