import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Pre-computation
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # For each block, compute which t-subsets it covers
    block_subsets = [None] * n_blocks
    for b_idx, block in enumerate(k_blocks):
        subs = []
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            subs.append(idx)
        block_subsets[b_idx] = subs
    
    # Build reverse index: for each t-subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx in range(n_blocks):
        for s in block_subsets[b_idx]:
            subset_to_blocks[s].append(b_idx)
    
    def criticality_greedy(seed=0, randomize=False):
        """Criticality-prioritized greedy: pick most constrained uncovered subset, then best block."""
        rng = random.Random(seed)
        uncovered = [True] * n_subsets
        n_uncovered = n_subsets
        selected = []
        selected_set = set()
        cover_count = [0] * n_subsets
        # For each subset, count how many blocks still available cover it
        # We'll track this implicitly by checking on the fly
        
        while n_uncovered > 0:
            if time.time() > deadline - 15:
                break
            
            # Find the most constrained uncovered subset (fewest covering blocks)
            best_subset = -1
            min_candidates = float('inf')
            
            # Sample some uncovered subsets for efficiency
            if n_uncovered > 500:
                sample_size = min(200, n_uncovered)
                candidates_sub = rng.sample([i for i in range(n_subsets) if uncovered[i]], sample_size)
            else:
                candidates_sub = [i for i in range(n_subsets) if uncovered[i]]
            
            for s in candidates_sub:
                if not uncovered[s]:
                    continue
                # Count how many unselected blocks cover this subset
                count = 0
                for b in subset_to_blocks[s]:
                    if b not in selected_set:
                        count += 1
                if count < min_candidates:
                    min_candidates = count
                    best_subset = s
                if count <= 1:
                    break
            
            if best_subset == -1:
                break
            
            # Find the best block covering this subset
            best_block = -1
            best_gain = 0
            
            # Get candidate blocks that cover this subset
            candidate_blocks = [b for b in subset_to_blocks[best_subset] if b not in selected_set]
            
            if not candidate_blocks:
                break
            
            if randomize:
                # Score each candidate by marginal gain
                scored = []
                for b in candidate_blocks:
                    gain = sum(1 for s2 in block_subsets[b] if uncovered[s2])
                    scored.append((gain, b))
                
                if not scored:
                    break
                
                scored.sort(reverse=True)
                n_consider = min(len(scored), max(10, len(scored) // 5))
                top_candidates = scored[:n_consider]
                
                # Weighted random selection
                weights = [g[0] ** 2 for g in top_candidates]
                total_w = sum(weights)
                if total_w == 0:
                    best_block = top_candidates[0][1]
                else:
                    r = rng.random() * total_w
                    cum = 0
                    chosen = top_candidates[0]
                    for g in top_candidates:
                        cum += weights[top_candidates.index(g)]
                        if r <= cum:
                            chosen = g
                            break
                    best_block = chosen[1]
            else:
                # Deterministic: pick block with max gain
                for b in candidate_blocks:
                    gain = sum(1 for s2 in block_subsets[b] if uncovered[s2])
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
            
            if best_block == -1:
                break
            
            selected.append(best_block)
            selected_set.add(best_block)
            for s in block_subsets[best_block]:
                if uncovered[s]:
                    uncovered[s] = False
                    n_uncovered -= 1
                cover_count[s] += 1
        
        return selected, selected_set, cover_count
    
    def remove_redundant(selected, selected_set, cover_count):
        """Remove redundant blocks."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                b_idx = selected[i]
                redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    selected_set.discard(b_idx)
                    changed = True
        return selected, selected_set, cover_count
    
    def local_search_1swap(selected, selected_set, cover_count, max_iters=100):
        """1-swap local search: try removing one block and adding another."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            
            for i in range(len(selected)):
                if time.time() > deadline - 3:
                    return selected, selected_set, cover_count
                
                b_remove = selected[i]
                
                # Find subsets that would become uncovered
                critical = [s for s in block_subsets[b_remove] if cover_count[s] == 1]
                
                if not critical:
                    # Redundant, just remove
                    for s in block_subsets[b_remove]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    selected_set.discard(b_remove)
                    improved = True
                    break
                
                # Find replacement: must cover all critical subsets
                best_add = -1
                best_score = float('-inf')
                
                # Sample candidates for speed
                candidates = [b for b in range(n_blocks) if b not in selected_set]
                if len(candidates) > 3000:
                    random.shuffle(candidates)
                    candidates = candidates[:3000]
                
                for b_add in candidates:
                    # Check if covers all critical
                    add_subs_set = set(block_subsets[b_add])
                    if not all(s in add_subs_set for s in critical):
                        continue
                    
                    # Score: prefer blocks that help barely-covered subsets
                    score = 0
                    for s in block_subsets[b_add]:
                        if cover_count[s] == 1:
                            score += 3
                        elif cover_count[s] == 2:
                            score += 1
                        else:
                            score -= 1
                    
                    if score > best_score:
                        best_score = score
                        best_add = b_add
                
                if best_add != -1:
                    # Perform swap
                    for s in block_subsets[b_remove]:
                        cover_count[s] -= 1
                    for s in block_subsets[best_add]:
                        cover_count[s] += 1
                    selected[i] = best_add
                    selected_set.discard(b_remove)
                    selected_set.add(best_add)
                    improved = True
                    break
        
        return selected, selected_set, cover_count
    
    def vns_2swap(selected, selected_set, cover_count, max_iters=50):
        """2-swap VNS: try removing 2 blocks and adding 2 different blocks."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2:
            improved = False
            iters += 1
            
            n_sel = len(selected)
            if n_sel < 2:
                break
            
            # Try pairs of blocks to remove
            for i in range(n_sel):
                if time.time() > deadline - 2:
                    return selected, selected_set, cover_count
                for j in range(i + 1, n_sel):
                    if time.time() > deadline - 2:
                        return selected, selected_set, cover_count
                    
                    b1 = selected[i]
                    b2 = selected[j]
                    
                    # Find subsets that would become uncovered
                    temp_cover = cover_count.copy()
                    for s in block_subsets[b1]:
                        temp_cover[s] -= 1
                    for s in block_subsets[b2]:
                        temp_cover[s] -= 1
                    
                    critical = [s for s in set(block_subsets[b1] + block_subsets[b2]) if temp_cover[s] == 0]
                    
                    if not critical:
                        # Both redundant, remove them
                        for s in block_subsets[b1]:
                            cover_count[s] -= 1
                        for s in block_subsets[b2]:
                            cover_count[s] -= 1
                        selected.pop(j)
                        selected.pop(i)
                        selected_set.discard(b1)
                        selected_set.discard(b2)
                        improved = True
                        break
                    
                    # Find 2 replacement blocks that cover all critical
                    best_pair = None
                    best_score = float('-inf')
                    
                    candidates = [b for b in range(n_blocks) if b not in selected_set and b != b1 and b != b2]
                    if len(candidates) > 2000:
                        random.shuffle(candidates)
                        candidates = candidates[:2000]
                    
                    # Sample pairs
                    n_cand = len(candidates)
                    n_pairs = min(500, n_cand * (n_cand - 1) // 2)
                    for _ in range(n_pairs):
                        idx1, idx2 = random.sample(range(n_cand), 2)
                        b_add1 = candidates[idx1]
                        b_add2 = candidates[idx2]
                        
                        add_subs = set(block_subsets[b_add1] + block_subsets[b_add2])
                        if not all(s in add_subs for s in critical):
                            continue
                        
                        score = 0
                        for s in add_subs:
                            if temp_cover[s] == 0:
                                score += 4
                            elif temp_cover[s] == 1:
                                score += 2
                            elif temp_cover[s] == 2:
                                score += 1
                            else:
                                score -= 1
                        
                        if score > best_score:
                            best_score = score
                            best_pair = (b_add1, b_add2)
                    
                    if best_pair is not None:
                        b_add1, b_add2 = best_pair
                        # Perform 2-swap
                        for s in block_subsets[b1]:
                            cover_count[s] -= 1
                        for s in block_subsets[b2]:
                            cover_count[s] -= 1
                        for s in block_subsets[b_add1]:
                            cover_count[s] += 1
                        for s in block_subsets[b_add2]:
                            cover_count[s] += 1
                        selected[i] = b_add1
                        selected[j] = b_add2
                        selected_set.discard(b1)
                        selected_set.discard(b2)
                        selected_set.add(b_add1)
                        selected_set.add(b_add2)
                        improved = True
                        break
                if improved:
                    break
        
        return selected, selected_set, cover_count
    
    def vns_3swap(selected, selected_set, cover_count, max_iters=20):
        """3-swap VNS: try removing 3 blocks and adding 3 different blocks."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2:
            improved = False
            iters += 1
            
            n_sel = len(selected)
            if n_sel < 3:
                break
            
            # Sample triples to remove
            n_attempts = 20
            for _ in range(n_attempts):
                if time.time() > deadline - 2:
                    return selected, selected_set, cover_count
                
                idxs = random.sample(range(n_sel), 3)
                b1, b2, b3 = selected[idxs[0]], selected[idxs[1]], selected[idxs[2]]
                
                temp_cover = cover_count.copy()
                for s in block_subsets[b1]:
                    temp_cover[s] -= 1
                for s in block_subsets[b2]:
                    temp_cover[s] -= 1
                for s in block_subsets[b3]:
                    temp_cover[s] -= 1
                
                critical = [s for s in set(block_subsets[b1] + block_subsets[b2] + block_subsets[b3]) if temp_cover[s] == 0]
                
                if not critical:
                    # All redundant
                    for b in [b1, b2, b3]:
                        for s in block_subsets[b]:
                            cover_count[s] -= 1
                        selected_set.discard(b)
                    for idx in sorted(idxs, reverse=True):
                        selected.pop(idx)
                    improved = True
                    break
                
                # Find 3 replacement blocks
                best_triple = None
                best_score = float('-inf')
                
                candidates = [b for b in range(n_blocks) if b not in selected_set and b not in (b1, b2, b3)]
                if len(candidates) > 1500:
                    random.shuffle(candidates)
                    candidates = candidates[:1500]
                
                n_cand = len(candidates)
                n_triples = min(200, n_cand * (n_cand - 1) * (n_cand - 2) // 6)
                for _ in range(n_triples):
                    idxs_add = random.sample(range(n_cand), 3)
                    b_add1, b_add2, b_add3 = candidates[idxs_add[0]], candidates[idxs_add[1]], candidates[idxs_add[2]]
                    
                    add_subs = set(block_subsets[b_add1] + block_subsets[b_add2] + block_subsets[b_add3])
                    if not all(s in add_subs for s in critical):
                        continue
                    
                    score = 0
                    for s in add_subs:
                        if temp_cover[s] == 0:
                            score += 5
                        elif temp_cover[s] == 1:
                            score += 2
                        elif temp_cover[s] == 2:
                            score += 1
                        else:
                            score -= 1
                    
                    if score > best_score:
                        best_score = score
                        best_triple = (b_add1, b_add2, b_add3)
                
                if best_triple is not None:
                    b_add1, b_add2, b_add3 = best_triple
                    for b in [b1, b2, b3]:
                        for s in block_subsets[b]:
                            cover_count[s] -= 1
                        selected_set.discard(b)
                    for b in [b_add1, b_add2, b_add3]:
                        for s in block_subsets[b]:
                            cover_count[s] += 1
                        selected_set.add(b)
                    for idx, b_new in zip(sorted(idxs, reverse=True), [b_add1, b_add2, b_add3]):
                        selected[idx] = b_new
                    improved = True
                    break
        
        return selected, selected_set, cover_count
    
    # Main search: multiple restarts with different strategies
    best_solution = None
    best_count = float('inf')
    
    # Deterministic criticality greedy
    if time.time() < deadline - 30:
        selected, selected_set, cover_count = criticality_greedy(seed=0, randomize=False)
        selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
        selected, selected_set, cover_count = local_search_1swap(selected, selected_set, cover_count, max_iters=200)
        selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_solution = selected[:]
        
        # VNS refinement
        selected, selected_set, cover_count = vns_2swap(selected, selected_set, cover_count, max_iters=50)
        selected, selected_set, cover_count = local_search_1swap(selected, selected_set, cover_count, max_iters=100)
        selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_solution = selected[:]
        
        selected, selected_set, cover_count = vns_3swap(selected, selected_set, cover_count, max_iters=20)
        selected, selected_set, cover_count = local_search_1swap(selected, selected_set, cover_count, max_iters=100)
        selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_solution = selected[:]
    
    # Randomized criticality greedy restarts
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        seed = r * 1000 + 42
        selected, selected_set, cover_count = criticality_greedy(seed=seed, randomize=True)
        selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
        selected, selected_set, cover_count = local_search_1swap(selected, selected_set, cover_count, max_iters=150)
        selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_solution = selected[:]
        
        # VNS on best
        if time.time() < deadline - 5:
            selected, selected_set, cover_count = vns_2swap(selected, selected_set, cover_count, max_iters=30)
            selected, selected_set, cover_count = local_search_1swap(selected, selected_set, cover_count, max_iters=100)
            selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
            
            if len(selected) < best_count:
                best_count = len(selected)
                best_solution = selected[:]
    
    # Final polish on best solution
    if best_solution is not None and time.time() < deadline - 2:
        # Rebuild cover_count
        cover_count = [0] * n_subsets
        selected_set = set(best_solution)
        for b_idx in best_solution:
            for s in block_subsets[b_idx]:
                cover_count[s] += 1
        
        # Final 1-swap pass
        selected = best_solution[:]
        selected, selected_set, cover_count = local_search_1swap(selected, selected_set, cover_count, max_iters=200)
        selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_solution = selected[:]
    
    if best_solution is None:
        # Fallback: simple greedy
        selected, selected_set, cover_count = criticality_greedy(seed=0, randomize=False)
        best_solution = selected
    
    return [list(k_blocks[i]) for i in best_solution]
