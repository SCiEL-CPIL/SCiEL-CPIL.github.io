import itertools
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

_PRECOMPUTED = {}

def _precompute(v, k, t):
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]
        
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build block -> tset index mapping and tset -> block indices
    block_tsets = []
    tset_to_block_indices = [[] for _ in range(n_tsets)]
    for b_idx, block in enumerate(all_blocks):
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(tset_indices)
        for ts_idx in tset_indices:
            tset_to_block_indices[ts_idx].append(b_idx)
    
    data = {
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'n_tsets': n_tsets,
        'block_tsets': block_tsets,
        'tset_to_block_indices': tset_to_block_indices,
        'tset_to_idx': tset_to_idx,
        'all_tsets': all_tsets
    }
    _PRECOMPUTED[(v, k, t)] = data
    return data

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    n_blocks = data['n_blocks']
    n_tsets = data['n_tsets']
    block_tsets = data['block_tsets']
    tset_to_block_indices = data['tset_to_block_indices']
    
    def _greedy_covering(rng, time_limit):
        """Fast greedy set cover with randomization."""
        deadline_greedy = time.time() + time_limit
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks_chosen = []
        
        while np.any(uncovered) and time.time() < deadline_greedy:
            # Find uncovered t-sets
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered t-set
            rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            
            # Find best block covering this t-set (maximizing total uncovered coverage)
            best_block = -1
            best_gain = -1
            for b_idx in tset_to_block_indices[rand_idx]:
                gain = 0
                for ts_idx in block_tsets[b_idx]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1 or best_gain <= 0:
                break
                
            blocks_chosen.append(best_block)
            
            # Update uncovered
            for ts_idx in block_tsets[best_block]:
                uncovered[ts_idx] = False
                
        return blocks_chosen
    
    def _local_search(blocks_idx, rng, time_limit):
        """Local search: remove redundant blocks and try swaps."""
        deadline_ls = time.time() + time_limit
        current = list(blocks_idx)
        
        # Precompute coverage for current solution
        covered = np.zeros(n_tsets, dtype=int)
        for b_idx in current:
            for ts_idx in block_tsets[b_idx]:
                covered[ts_idx] += 1
        
        best = current[:]
        best_uncovered = n_tsets - np.count_nonzero(covered)
        
        while time.time() < deadline_ls:
            improved = False
            
            # Try removing redundant blocks
            for i in range(len(current)):
                if time.time() > deadline_ls:
                    break
                b_idx = current[i]
                redundant = True
                for ts_idx in block_tsets[b_idx]:
                    if covered[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    new_current = current[:]
                    del new_current[i]
                    for ts_idx in block_tsets[b_idx]:
                        covered[ts_idx] -= 1
                    current = new_current
                    uncovered_count = n_tsets - np.count_nonzero(covered)
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best = current[:]
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swapping: replace one block with another
            for i in range(len(current)):
                if time.time() > deadline_ls:
                    break
                old_b = current[i]
                newly_uncovered = []
                for ts_idx in block_tsets[old_b]:
                    if covered[ts_idx] == 1:
                        newly_uncovered.append(ts_idx)
                
                if not newly_uncovered:
                    current.pop(i)
                    for ts_idx in block_tsets[old_b]:
                        covered[ts_idx] -= 1
                    uncovered_count = n_tsets - np.count_nonzero(covered)
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best = current[:]
                    improved = True
                    break
                
                # Try to find a replacement block that covers all newly_uncovered
                best_new_b = -1
                best_gain = -1
                for b_idx in tset_to_block_indices[newly_uncovered[0]]:
                    if b_idx == old_b or b_idx in current:
                        continue
                    covers_all = True
                    for ts_idx in newly_uncovered:
                        if ts_idx not in block_tsets[b_idx]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    gain = 0
                    for ts_idx in block_tsets[b_idx]:
                        if covered[ts_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_new_b = b_idx
                
                if best_new_b != -1:
                    current[i] = best_new_b
                    for ts_idx in block_tsets[old_b]:
                        covered[ts_idx] -= 1
                    for ts_idx in block_tsets[best_new_b]:
                        covered[ts_idx] += 1
                    uncovered_count = n_tsets - np.count_nonzero(covered)
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best = current[:]
                    improved = True
                    break
            
            if not improved:
                # Try random perturbation
                if time.time() > deadline_ls:
                    break
                for _ in range(30):
                    if time.time() > deadline_ls:
                        break
                    if len(current) < 2:
                        break
                    i, j = rng.sample(range(len(current)), 2)
                    old_i, old_j = current[i], current[j]
                    
                    for ts_idx in block_tsets[old_i]:
                        covered[ts_idx] -= 1
                    for ts_idx in block_tsets[old_j]:
                        covered[ts_idx] -= 1
                    
                    new_i, new_j = rng.sample(range(n_blocks), 2)
                    current[i] = new_i
                    current[j] = new_j
                    for ts_idx in block_tsets[new_i]:
                        covered[ts_idx] += 1
                    for ts_idx in block_tsets[new_j]:
                        covered[ts_idx] += 1
                    
                    uncovered_count = n_tsets - np.count_nonzero(covered)
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best = current[:]
    
        return best
    
    def _vns_search(blocks_idx, rng, time_limit):
        """Variable Neighborhood Search with nested neighborhoods."""
        deadline_vns = time.time() + time_limit
        current = list(blocks_idx)
        
        # Precompute coverage
        covered = np.zeros(n_tsets, dtype=int)
        for b_idx in current:
            for ts_idx in block_tsets[b_idx]:
                covered[ts_idx] += 1
        
        best = current[:]
        best_size = len(current)
        
        def _get_uncovered_count(covered_arr):
            return n_tsets - np.count_nonzero(covered_arr)
        
        def _is_valid(covered_arr):
            return np.all(covered_arr > 0)
        
        # Neighborhood sizes to try
        r_values = [1, 2, 3]
        
        while time.time() < deadline_vns:
            found_improvement = False
            
            for r in r_values:
                if time.time() > deadline_vns:
                    break
                
                # Try all combinations of r blocks to swap
                n_curr = len(current)
                if n_curr < r:
                    continue
                
                # For small r, enumerate all r-subsets
                if r <= 2:
                    if r == 1:
                        subsets = [(i,) for i in range(n_curr)]
                    else:
                        subsets = list(itertools.combinations(range(n_curr), r))
                else:
                    # For r=3, sample a subset of combinations
                    all_combos = list(itertools.combinations(range(n_curr), r))
                    if len(all_combos) > 5000:
                        rng.shuffle(all_combos)
                        all_combos = all_combos[:5000]
                    subsets = all_combos
                
                for combo in subsets:
                    if time.time() > deadline_vns:
                        break
                    
                    old_blocks = [current[i] for i in combo]
                    
                    # Temporarily remove these blocks
                    for b_idx in old_blocks:
                        for ts_idx in block_tsets[b_idx]:
                            covered[ts_idx] -= 1
                    
                    # Find which t-sets are now uncovered
                    newly_uncovered = [ts_idx for ts_idx in range(n_tsets) if covered[ts_idx] == 0]
                    
                    if len(newly_uncovered) > r * 35:
                        # Too many uncovered, unlikely to fix with r new blocks
                        for b_idx in old_blocks:
                            for ts_idx in block_tsets[b_idx]:
                                covered[ts_idx] += 1
                        continue
                    
                    # Try to find r blocks that cover all newly_uncovered
                    if not newly_uncovered:
                        # No new uncovered, just remove the old blocks
                        new_current = [b for i, b in enumerate(current) if i not in combo]
                        new_size = len(new_current)
                        if new_size < best_size and _is_valid(covered):
                            best_size = new_size
                            best = new_current[:]
                            current = new_current[:]
                            found_improvement = True
                        for b_idx in old_blocks:
                            for ts_idx in block_tsets[b_idx]:
                                covered[ts_idx] += 1
                        if found_improvement:
                            break
                        continue
                    
                    # Find candidate blocks that cover at least one newly_uncovered t-set
                    candidates = set()
                    for ts_idx in newly_uncovered:
                        for b_idx in tset_to_block_indices[ts_idx]:
                            if b_idx not in old_blocks:
                                candidates.add(b_idx)
                    
                    if len(candidates) < r:
                        for b_idx in old_blocks:
                            for ts_idx in block_tsets[b_idx]:
                                covered[ts_idx] += 1
                        continue
                    
                    # Try to find a set of r blocks that covers all newly_uncovered
                    candidate_list = list(candidates)
                    rng.shuffle(candidate_list)
                    
                    best_combo_found = None
                    best_gain = -1
                    
                    if r == 1:
                        for b_idx in candidate_list[:200]:
                            # Check if this single block covers all newly_uncovered
                            covers_all = all(ts_idx in block_tsets[b_idx] for ts_idx in newly_uncovered)
                            if covers_all:
                                # Count additional coverage
                                gain = 0
                                for ts_idx in block_tsets[b_idx]:
                                    if covered[ts_idx] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_combo_found = [b_idx]
                    else:
                        # For r > 1, use a simple heuristic
                        # Sort candidates by how many newly_uncovered they cover
                        scored_candidates = []
                        for b_idx in candidate_list[:500]:
                            count = sum(1 for ts_idx in newly_uncovered if ts_idx in block_tsets[b_idx])
                            scored_candidates.append((count, b_idx))
                        scored_candidates.sort(reverse=True)
                        
                        # Greedily pick r blocks
                        selected = []
                        remaining_uncovered = set(newly_uncovered)
                        for count, b_idx in scored_candidates:
                            if len(selected) >= r:
                                break
                            if not remaining_uncovered:
                                break
                            if b_idx in selected:
                                continue
                            # Check if adding this block helps
                            new_covered = 0
                            for ts_idx in remaining_uncovered:
                                if ts_idx in block_tsets[b_idx]:
                                    new_covered += 1
                            if new_covered > 0:
                                selected.append(b_idx)
                                remaining_uncovered -= block_tsets[b_idx]
                        
                        if len(selected) == r and not remaining_uncovered:
                            # Verify: do these r blocks cover all newly_uncovered?
                            all_covered = set()
                            for b_idx in selected:
                                all_covered |= block_tsets[b_idx]
                            if all(ts_idx in all_covered for ts_idx in newly_uncovered):
                                # Count additional coverage
                                gain = 0
                                for b_idx in selected:
                                    for ts_idx in block_tsets[b_idx]:
                                        if covered[ts_idx] == 0:
                                            gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_combo_found = selected
                    
                    # Restore coverage
                    for b_idx in old_blocks:
                        for ts_idx in block_tsets[b_idx]:
                            covered[ts_idx] += 1
                    
                    if best_combo_found is not None:
                        # Perform the swap
                        for i, b_idx in zip(combo, best_combo_found):
                            for ts_idx in block_tsets[current[i]]:
                                covered[ts_idx] -= 1
                            for ts_idx in block_tsets[b_idx]:
                                covered[ts_idx] += 1
                            current[i] = b_idx
                        
                        if _is_valid(covered) and len(current) < best_size:
                            best_size = len(current)
                            best = current[:]
                            found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                # Try a random shake
                if time.time() > deadline_vns:
                    break
                for _ in range(20):
                    if time.time() > deadline_vns:
                        break
                    if len(current) < 2:
                        break
                    i, j = rng.sample(range(len(current)), 2)
                    old_i, old_j = current[i], current[j]
                    
                    for ts_idx in block_tsets[old_i]:
                        covered[ts_idx] -= 1
                    for ts_idx in block_tsets[old_j]:
                        covered[ts_idx] -= 1
                    
                    new_i, new_j = rng.sample(range(n_blocks), 2)
                    current[i] = new_i
                    current[j] = new_j
                    for ts_idx in block_tsets[new_i]:
                        covered[ts_idx] += 1
                    for ts_idx in block_tsets[new_j]:
                        covered[ts_idx] += 1
                    
                    if _is_valid(covered) and len(current) < best_size:
                        best_size = len(current)
                        best = current[:]
    
        return best
    
    # Multiple greedy runs with different seeds
    best_blocks = None
    best_size = float('inf')
    
    for seed in range(10):
        if time.time() > deadline - 40.0:
            break
        rng = random.Random(seed)
        blocks_idx = _greedy_covering(rng, 4.0)
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            best_blocks = blocks_idx[:]
    
    if best_blocks is None:
        rng = random.Random(42)
        best_blocks = _greedy_covering(rng, 8.0)
        best_size = len(best_blocks)
    
    # Local search on best solution
    if time.time() < deadline - 30.0:
        rng = random.Random(123)
        remaining_time = min(40.0, deadline - (time.time() - start_time) - 30.0)
        improved = _local_search(best_blocks, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # VNS search
    if time.time() < deadline - 15.0 and best_size > 70:
        rng = random.Random(456)
        remaining_time = min(25.0, deadline - (time.time() - start_time) - 15.0)
        improved = _vns_search(best_blocks, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        rng = random.Random(789)
        remaining_time = min(10.0, deadline - (time.time() - start_time) - 5.0)
        improved = _local_search(best_blocks, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Final verification
    covered_final = np.zeros(n_tsets, dtype=int)
    for b_idx in best_blocks:
        for ts_idx in block_tsets[b_idx]:
            covered_final[ts_idx] += 1
    
    if not np.all(covered_final > 0):
        # Rebuild with a fresh greedy
        rng = random.Random(1000)
        best_blocks = _greedy_covering(rng, 10.0)
        best_size = len(best_blocks)
    
    # Convert to list of lists
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result

# EVOLVE-BLOCK-END
