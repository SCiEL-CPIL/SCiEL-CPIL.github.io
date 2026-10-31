import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0
    
    # Fallback for non-standard parameters
    if v != 19 or k != 7 or t != 4:
        return _greedy_generic(v, k, t)
    
    # --- Precomputation ---
    # 1. Map 4-subsets to IDs
    map4 = {}
    id_counter = 0
    for comb in itertools.combinations(range(v), t):
        map4[comb] = id_counter
        id_counter += 1
    n_4s = id_counter  # 3876
    
    # 2. Generate candidate blocks
    # We generate a diverse set of candidates.
    # For v=19, k=7, t=4, we need enough candidates to ensure coverage.
    # We use random sampling + cyclic shifts to ensure diversity.
    
    random.seed(42)
    candidate_blocks = set()
    
    # Add cyclic orbits of random base blocks
    # This ensures symmetry and good coverage properties
    n_orbits = 200
    for _ in range(n_orbits):
        base = tuple(sorted(random.sample(range(v), k)))
        # Normalize to canonical form (min element 0) to identify orbit
        # But we add all shifts to the candidate pool
        for shift in range(v):
            shifted = tuple(sorted((b + shift) % v for b in base))
            candidate_blocks.add(shifted)
    
    # Add some purely random blocks to break symmetry
    for _ in range(500):
        block = tuple(sorted(random.sample(range(v), k)))
        candidate_blocks.add(block)
    
    candidate_blocks = sorted(candidate_blocks)
    n_candidates = len(candidate_blocks)
    
    # 3. Pre-compute coverage for each candidate block
    # block_4s[i] is the list of 4-subset IDs covered by candidate block i
    block_4s = []
    for block in candidate_blocks:
        ids = [map4[sub] for sub in itertools.combinations(block, t)]
        block_4s.append(ids)
        
    # 4. Invert: for each 4-subset, which candidate blocks cover it?
    sub_blocks = [[] for _ in range(n_4s)]
    for b_idx, ids in enumerate(block_4s):
        for sid in ids:
            sub_blocks[sid].append(b_idx)
            
    # --- Greedy Construction ---
    def greedy_construct(seed_offset=0):
        random.seed(42 + seed_offset)
        counts = np.zeros(n_4s, dtype=np.int32)
        selected_blocks = []
        uncovered_mask = np.ones(n_4s, dtype=bool)
        uncovered_count = n_4s
        
        while uncovered_count > 0:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Pick a random uncovered subset
            unc_idx = np.where(uncovered_mask)[0]
            target_sid = unc_idx[random.randint(0, len(unc_idx) - 1)]
            
            # Get candidates covering this target
            candidates = sub_blocks[target_sid]
            
            if not candidates:
                break
                
            best_cand_score = -1
            best_cand_idx = -1
            
            for c_idx in candidates:
                # Count how many uncovered subsets this candidate covers
                score = 0
                for sid in block_4s[c_idx]:
                    if uncovered_mask[sid]:
                        score += 1
                        if score > best_cand_score:
                            best_cand_score = score
                            best_cand_idx = c_idx
                            if score == 35: # Max
                                break
                
            if best_cand_idx == -1:
                break
                
            # Select this block
            selected_blocks.append(best_cand_idx)
            
            # Update counts and uncovered mask
            for sid in block_4s[best_cand_idx]:
                if uncovered_mask[sid]:
                    uncovered_mask[sid] = False
                    uncovered_count -= 1
            
            # Check time
            if time.time() - start_time > time_limit - 5:
                break
        
        return selected_blocks, uncovered_count

    # --- Local Search ---
    def local_search(initial_blocks, time_budget):
        current_blocks = list(initial_blocks)
        counts = np.zeros(n_4s, dtype=np.int32)
        for b_idx in current_blocks:
            for sid in block_4s[b_idx]:
                counts[sid] += 1
        
        # Verify validity
        if np.any(counts == 0):
             uncovered_mask = (counts == 0)
             while np.any(uncovered_mask):
                 if time.time() - start_time > time_limit - 5:
                     break
                 unc_idx = np.where(uncovered_mask)[0]
                 target_sid = unc_idx[0]
                 candidates = sub_blocks[target_sid]
                 if not candidates:
                     break
                 best_cand = -1
                 best_score = -1
                 for c_idx in candidates:
                     score = 0
                     for sid in block_4s[c_idx]:
                         if uncovered_mask[sid]:
                             score += 1
                     if score > best_score:
                         best_score = score
                         best_cand = c_idx
                 if best_cand == -1:
                     break
                 current_blocks.append(best_cand)
                 for sid in block_4s[best_cand]:
                     counts[sid] += 1
                     if counts[sid] == 1:
                         uncovered_mask[sid] = False
                 uncovered_mask = (counts == 0)
        
        best_len = len(current_blocks)
        no_improve = 0
        start_ls = time.time()
        
        while time.time() - start_ls < time_budget and no_improve < 10:
            improved = False
            
            # 1. Try removing a block
            if current_blocks:
                remove_candidates = []
                for b_idx in current_blocks:
                    unique_count = 0
                    for sid in block_4s[b_idx]:
                        if counts[sid] == 1:
                            unique_count += 1
                    remove_candidates.append((unique_count, b_idx))
                
                remove_candidates.sort()
                _, b_to_remove = remove_candidates[0]
                
                safe = True
                for sid in block_4s[b_to_remove]:
                    if counts[sid] == 1:
                        safe = False
                        break
                
                if safe:
                    current_blocks.remove(b_to_remove)
                    for sid in block_4s[b_to_remove]:
                        counts[sid] -= 1
                    improved = True
                    no_improve = 0
                    continue
                else:
                    no_improve += 1
            
            # 2. Try 1-swap: Remove one, add one
            if current_blocks and time.time() - start_time < time_limit - 5:
                b_old = random.choice(current_blocks)
                for sid in block_4s[b_old]:
                    counts[sid] -= 1
                
                uncovered_now = (counts == 0)
                
                if np.any(uncovered_now):
                    unc_idx = np.where(uncovered_now)[0]
                    target_sid = unc_idx[0]
                    candidates = sub_blocks[target_sid]
                    
                    best_cand = -1
                    best_score = -1
                    for c_idx in candidates:
                        if c_idx == b_old:
                            continue
                        if c_idx in current_blocks:
                            continue
                        score = 0
                        for sid in block_4s[c_idx]:
                            if uncovered_now[sid]:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_cand = c_idx
                    
                    if best_cand != -1:
                        current_blocks.remove(b_old)
                        current_blocks.append(best_cand)
                        for sid in block_4s[best_cand]:
                            counts[sid] += 1
                        if not np.any(counts == 0):
                            improved = True
                            no_improve = 0
                        else:
                            current_blocks.remove(best_cand)
                            current_blocks.append(b_old)
                            for sid in block_4s[best_cand]:
                                counts[sid] -= 1
                            for sid in block_4s[b_old]:
                                counts[sid] += 1
                            no_improve += 1
                else:
                    current_blocks.remove(b_old)
                    improved = True
                    no_improve = 0
        
        return current_blocks

    # --- Main Loop ---
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Greedy phase
        init_blocks, unc_count = greedy_construct(i)
        
        if unc_count > 0:
            counts = np.zeros(n_4s, dtype=np.int32)
            for b_idx in init_blocks:
                for sid in block_4s[b_idx]:
                    counts[sid] += 1
            uncovered_mask = (counts == 0)
            while np.any(uncovered_mask):
                if time.time() - start_time > time_limit - 5:
                    break
                unc_idx = np.where(uncovered_mask)[0]
                target_sid = unc_idx[0]
                candidates = sub_blocks[target_sid]
                if not candidates:
                    break
                c_idx = random.choice(candidates)
                if c_idx not in init_blocks:
                    init_blocks.append(c_idx)
                    for sid in block_4s[c_idx]:
                        counts[sid] += 1
                    uncovered_mask = (counts == 0)
        
        # Local search phase
        elapsed = time.time() - start_time
        remaining = time_limit - 10 - elapsed
        if remaining < 1:
            break
            
        ls_blocks = local_search(init_blocks, remaining)
        
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Fallback if no valid solution found
    if best_blocks is None:
        counts = np.zeros(n_4s, dtype=np.int32)
        best_blocks = []
        uncovered_mask = np.ones(n_4s, dtype=bool)
        while np.any(uncovered_mask):
            if time.time() - start_time > time_limit - 2:
                break
            unc_idx = np.where(uncovered_mask)[0]
            target_sid = unc_idx[0]
            candidates = sub_blocks[target_sid]
            if not candidates:
                break
            c_idx = random.choice(candidates)
            if c_idx not in best_blocks:
                best_blocks.append(c_idx)
                for sid in block_4s[c_idx]:
                    counts[sid] += 1
                uncovered_mask = (counts == 0)
    
    # Final validity check and repair
    counts = np.zeros(n_4s, dtype=np.int32)
    for b_idx in best_blocks:
        for sid in block_4s[b_idx]:
            counts[sid] += 1
    
    uncovered_mask = (counts == 0)
    while np.any(uncovered_mask):
        if time.time() - start_time > time_limit - 1:
            break
        unc_idx = np.where(uncovered_mask)[0]
        target_sid = unc_idx[0]
        candidates = sub_blocks[target_sid]
        if not candidates:
            break
        c_idx = random.choice(candidates)
        if c_idx not in best_blocks:
            best_blocks.append(c_idx)
            for sid in block_4s[c_idx]:
                counts[sid] += 1
            uncovered_mask = (counts == 0)
    
    result = [list(candidate_blocks[i]) for i in best_blocks]
    return result

def _greedy_generic(v, k, t):
    # Fallback for non-19,7,4
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
