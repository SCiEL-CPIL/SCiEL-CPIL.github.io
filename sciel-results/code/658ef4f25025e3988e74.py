"""Simulated annealing for covering design C(24,7,3)."""
import itertools
import random
import math
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    # Precompute all t-subsets (triples) and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, which triples it covers
    block_triples = []
    for b in all_blocks:
        trs = tuple(triple_index[tr] for tr in itertools.combinations(b, t))
        block_triples.append(trs)
    block_triples_arr = [np.array(trs, dtype=np.int32) for trs in block_triples]
    
    # For each triple, which blocks contain it
    triple_to_blocks = [[] for _ in range(n_triples)]
    for b_idx, b in enumerate(all_blocks):
        for tr in itertools.combinations(b, t):
            tri_idx = triple_index[tr]
            triple_to_blocks[tri_idx].append(b_idx)
    
    # --- Step 1: Greedy initial covering ---
    uncovered = np.ones(n_triples, dtype=bool)
    blocks_chosen = set()
    
    while uncovered.any():
        if time.time() - start_time > time_limit * 0.3:
            break
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        tri_idx = unc_idx[random.randint(0, len(unc_idx)-1)]
        candidates = triple_to_blocks[tri_idx]
        best_b = -1
        best_cov = -1
        for b_idx in candidates:
            trs = block_triples_arr[b_idx]
            cov = np.sum(uncovered[trs])
            if cov > best_cov:
                best_cov = cov
                best_b = b_idx
        blocks_chosen.add(best_b)
        uncovered[block_triples_arr[best_b]] = False
    
    # --- Step 2: Delete redundant blocks ---
    coverage = np.zeros(n_triples, dtype=np.int32)
    for b_idx in blocks_chosen:
        coverage[block_triples_arr[b_idx]] += 1
    
    changed = True
    while changed and time.time() - start_time < time_limit * 0.4:
        changed = False
        for b_idx in list(blocks_chosen):
            trs = block_triples_arr[b_idx]
            if np.any(coverage[trs] == 1):
                continue
            blocks_chosen.remove(b_idx)
            coverage[trs] -= 1
            changed = True
    
    # --- Step 3: Simulated annealing ---
    current_blocks = set(blocks_chosen)
    coverage = np.zeros(n_triples, dtype=np.int32)
    for b_idx in current_blocks:
        coverage[block_triples_arr[b_idx]] += 1
    
    uncovered_count = int(np.sum(coverage == 0))
    block_count = len(current_blocks)
    
    best_blocks = set(current_blocks)
    best_block_count = block_count
    best_energy = uncovered_count * 1000 + block_count
    current_energy = best_energy
    
    T = 5.0
    cooling = 0.998
    min_T = 0.05
    
    while time.time() - start_time < time_limit * 0.9 and T > min_T:
        move_type = random.random()
        if move_type < 0.35:
            # Remove a random block
            if len(current_blocks) == 0:
                T *= cooling
                continue
            b_idx = random.choice(list(current_blocks))
            trs = block_triples_arr[b_idx]
            new_cov = coverage[trs] - 1
            new_unc = uncovered_count + int(np.sum(new_cov == 0))
            new_bc = block_count - 1
            new_energy = new_unc * 1000 + new_bc
            delta = new_energy - current_energy
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_blocks.remove(b_idx)
                coverage[trs] -= 1
                uncovered_count = new_unc
                block_count = new_bc
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = set(current_blocks)
                    best_block_count = block_count
        elif move_type < 0.65:
            # Add a random block
            b_idx = random.randint(0, n_blocks - 1)
            if b_idx in current_blocks:
                T *= cooling
                continue
            trs = block_triples_arr[b_idx]
            new_cov = coverage[trs] + 1
            new_unc = uncovered_count - int(np.sum((new_cov == 1) & (coverage[trs] == 0)))
            new_bc = block_count + 1
            new_energy = new_unc * 1000 + new_bc
            delta = new_energy - current_energy
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_blocks.add(b_idx)
                coverage[trs] += 1
                uncovered_count = new_unc
                block_count = new_bc
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = set(current_blocks)
                    best_block_count = block_count
        else:
            # Swap
            if len(current_blocks) == 0:
                T *= cooling
                continue
            b_remove = random.choice(list(current_blocks))
            b_add = random.randint(0, n_blocks - 1)
            if b_add == b_remove or b_add in current_blocks:
                T *= cooling
                continue
            trs_rem = block_triples_arr[b_remove]
            trs_add = block_triples_arr[b_add]
            affected = np.union1d(trs_rem, trs_add)
            new_unc = uncovered_count
            for tri in affected:
                old_c = coverage[tri]
                in_rem = tri in trs_rem
                in_add = tri in trs_add
                if in_rem and in_add:
                    new_c = old_c
                elif in_rem:
                    new_c = old_c - 1
                else:
                    new_c = old_c + 1
                if old_c == 0 and new_c > 0:
                    new_unc -= 1
                elif old_c > 0 and new_c == 0:
                    new_unc += 1
            new_bc = block_count
            new_energy = new_unc * 1000 + new_bc
            delta = new_energy - current_energy
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_blocks.remove(b_remove)
                current_blocks.add(b_add)
                coverage[trs_rem] -= 1
                coverage[trs_add] += 1
                uncovered_count = new_unc
                block_count = new_bc
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = set(current_blocks)
                    best_block_count = block_count
        
        T *= cooling
    
    # Ensure validity: if best_blocks has uncovered triples, fall back to a valid covering
    # Check if best_blocks covers all triples
    cov_check = np.zeros(n_triples, dtype=np.int32)
    for b_idx in best_blocks:
        cov_check[block_triples_arr[b_idx]] += 1
    if np.any(cov_check == 0):
        # Fall back to current_blocks if it's valid, else rebuild
        cov_cur = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_blocks:
            cov_cur[block_triples_arr[b_idx]] += 1
        if np.any(cov_cur == 0):
            # Rebuild a valid covering from scratch (greedy)
            uncovered = np.ones(n_triples, dtype=bool)
            valid_blocks = set()
            while uncovered.any() and time.time() - start_time < time_limit - 2:
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) == 0:
                    break
                tri_idx = unc_idx[random.randint(0, len(unc_idx)-1)]
                candidates = triple_to_blocks[tri_idx]
                best_b = -1
                best_cov = -1
                for b_idx in candidates:
                    trs = block_triples_arr[b_idx]
                    cov = np.sum(uncovered[trs])
                    if cov > best_cov:
                        best_cov = cov
                        best_b = b_idx
                valid_blocks.add(best_b)
                uncovered[block_triples_arr[best_b]] = False
            best_blocks = valid_blocks
    
    result = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return result
