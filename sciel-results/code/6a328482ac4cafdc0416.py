import itertools
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)
    
    # Precompute all t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-sets it covers
    block_tsets = []
    for block in all_blocks:
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(tset_indices)
    
    # Build tset_to_block_indices for fast lookup
    tset_to_block_indices = [[] for _ in range(n_tsets)]
    for b_idx in range(n_blocks):
        for ts_idx in block_tsets[b_idx]:
            tset_to_block_indices[ts_idx].append(b_idx)
    
    # PBD-inspired: generate structured block pool with uniform triple coverage
    def _generate_pbd_pool(rng, target_size=120):
        """Generate blocks approximating a 2-(24,7,λ) design for uniform triple coverage."""
        # Start with random blocks, then adjust for uniformity
        point_pair_counts = np.zeros((v, v), dtype=int)
        triple_counts = np.zeros(n_tsets, dtype=int)
        
        pool = []
        used_blocks = set()
        
        # First pass: random selection
        for _ in range(target_size):
            if time.time() > deadline - 60:
                break
            attempts = 0
            while attempts < 50:
                block = tuple(sorted(rng.sample(range(v), k)))
                if block not in used_blocks:
                    break
                attempts += 1
            else:
                continue
            
            used_blocks.add(block)
            pool.append(block)
            
            # Update counts
            for a, b in itertools.combinations(block, 2):
                point_pair_counts[a, b] += 1
                point_pair_counts[b, a] += 1
            for ts in itertools.combinations(block, t):
                triple_counts[tset_to_idx[ts]] += 1
        
        # Second pass: add blocks that cover under-represented triples
        for _ in range(target_size // 2):
            if time.time() > deadline - 50:
                break
            
            # Find most under-represented triple
            min_count = np.min(triple_counts)
            under_represented = np.where(triple_counts == min_count)[0]
            if len(under_represented) == 0:
                break
            
            # Pick a random under-represented triple
            target_ts_idx = under_represented[rng.randint(0, len(under_represented) - 1)]
            target_ts = all_tsets[target_ts_idx]
            
            # Find blocks containing this triple
            candidates = tset_to_block_indices[target_ts_idx]
            if not candidates:
                continue
            
            # Score candidates by how many under-represented triples they cover
            best_block = -1
            best_score = -1
            for b_idx in candidates[:50]:  # Limit candidates for speed
                if all_blocks[b_idx] in used_blocks:
                    continue
                score = 0
                for ts_idx in block_tsets[b_idx]:
                    score += (min_count + 5 - triple_counts[ts_idx])  # Weight under-represented
                if score > best_score:
                    best_score = score
                    best_block = b_idx
            
            if best_block != -1:
                block = all_blocks[best_block]
                if block not in used_blocks:
                    used_blocks.add(block)
                    pool.append(block)
                    for a, b in itertools.combinations(block, 2):
                        point_pair_counts[a, b] += 1
                        point_pair_counts[b, a] += 1
                    for ts in itertools.combinations(block, t):
                        triple_counts[tset_to_idx[ts]] += 1
        
        return [list(b) for b in pool]
    
    def _greedy_deletion(blocks_list, rng, time_limit):
        """Bitset-based greedy deletion: remove blocks that don't reduce coverage."""
        deadline_del = time.time() + time_limit
        
        # Convert to indices
        block_idx_map = {}
        for i, block in enumerate(blocks_list):
            block_idx_map[tuple(block)] = i
        
        current_idx = [block_idx_map[tuple(b)] for b in blocks_list]
        
        # Build coverage counts
        covered = np.zeros(n_tsets, dtype=int)
        for b_idx in current_idx:
            for ts_idx in block_tsets[b_idx]:
                covered[ts_idx] += 1
        
        improved = True
        while improved and time.time() < deadline_del:
            improved = False
            for i in range(len(current_idx)):
                if time.time() > deadline_del:
                    break
                b_idx = current_idx[i]
                # Check if any t-set would become uncovered
                redundant = True
                for ts_idx in block_tsets[b_idx]:
                    if covered[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for ts_idx in block_tsets[b_idx]:
                        covered[ts_idx] -= 1
                    del current_idx[i]
                    improved = True
                    break
        
        return [all_blocks[idx] for idx in current_idx]
    
    def _greedy_covering(rng, time_limit):
        """Fast greedy set cover."""
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
    
    # Strategy 1: PBD-inspired pool + greedy deletion
    best_blocks = None
    best_size = float('inf')
    
    if time.time() < deadline - 50:
        rng = random.Random(42)
        pbd_pool = _generate_pbd_pool(rng, target_size=100)
        if pbd_pool:
            # Convert to indices
            pbd_indices = []
            for block in pbd_pool:
                for i, b in enumerate(all_blocks):
                    if b == tuple(block):
                        pbd_indices.append(i)
                        break
            
            # Apply greedy deletion
            deleted = _greedy_deletion(pbd_pool, rng, 20)
            if deleted:
                # Verify coverage
                covered_check = np.zeros(n_tsets, dtype=int)
                for block in deleted:
                    for ts in itertools.combinations(block, t):
                        covered_check[tset_to_idx[ts]] += 1
                if np.all(covered_check > 0):
                    if len(deleted) < best_size:
                        best_size = len(deleted)
                        best_blocks = deleted[:]
    
    # Strategy 2: Multiple greedy runs
    for seed in range(8):
        if time.time() > deadline - 40.0:
            break
        rng = random.Random(seed * 100 + 7)
        blocks_idx = _greedy_covering(rng, 3.0)
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            # Convert to block lists
            best_blocks = [list(all_blocks[idx]) for idx in blocks_idx]
    
    if best_blocks is None:
        rng = random.Random(42)
        blocks_idx = _greedy_covering(rng, 8.0)
        best_blocks = [list(all_blocks[idx]) for idx in blocks_idx]
        best_size = len(best_blocks)
    
    # Local search on best solution
    if time.time() < deadline - 30.0:
        # Convert back to indices for local search
        block_idx_map = {}
        for i, block in enumerate(all_blocks):
            block_idx_map[block] = i
        
        best_indices = [block_idx_map[tuple(b)] for b in best_blocks]
        
        rng = random.Random(123)
        remaining_time = min(35.0, deadline - (time.time() - start_time) - 30.0)
        improved = _local_search(best_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = [list(all_blocks[idx]) for idx in improved]
            best_size = len(improved)
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        block_idx_map = {}
        for i, block in enumerate(all_blocks):
            block_idx_map[block] = i
        
        best_indices = [block_idx_map[tuple(b)] for b in best_blocks]
        
        rng = random.Random(789)
        remaining_time = min(8.0, deadline - (time.time() - start_time) - 5.0)
        improved = _local_search(best_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = [list(all_blocks[idx]) for idx in improved]
            best_size = len(improved)
    
    # Final verification
    covered_final = np.zeros(n_tsets, dtype=int)
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered_final[tset_to_idx[ts]] += 1
    
    if not np.all(covered_final > 0):
        # Rebuild with a fresh greedy
        rng = random.Random(1000)
        blocks_idx = _greedy_covering(rng, 10.0)
        best_blocks = [list(all_blocks[idx]) for idx in blocks_idx]
        best_size = len(best_blocks)
    
    return best_blocks

# EVOLVE-BLOCK-END
