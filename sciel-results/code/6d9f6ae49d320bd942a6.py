import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    best_triples = []
    best_alpha = n
    best_size = 0
    
    # Phase 1: Greedy Random Restart to find large packings
    # We want to maximize the number of triples first.
    
    def greedy_packing():
        used_pairs = set()
        triples = []
        # Shuffle candidates
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for tri in candidates:
            pairs = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                triples.append(list(tri))
                for p in pairs:
                    used_pairs.add(p)
        return triples

    # Run greedy restarts
    # Estimate time per restart: ~0.01s?
    # 110s / 0.01s = 11000 restarts.
    # We will keep the top 10 largest packings found.
    
    top_packings = [] # list of (size, triples)
    
    while time.time() < deadline - 30.0:
        pack = greedy_packing()
        size = len(pack)
        
        if size > best_size:
            best_size = size
            top_packings = [(size, pack)]
        elif size == best_size:
            top_packings.append((size, pack))
            
        # Keep top_packings manageable
        if len(top_packings) > 50:
            # Sort by size (all same if we only add same size? No, we might find smaller ones)
            # Actually, we only add if size >= best_size.
            # So top_packings contains only best_size packings.
            # We can just truncate.
            top_packings = top_packings[-20:]

    # Phase 2: Evaluate alpha for the best packings
    if not top_packings:
        # Fallback
        pack = greedy_packing()
        top_packings = [(len(pack), pack)]

    # Calculate alpha for the top packings
    for size, pack in top_packings:
        if time.time() > deadline - 10.0:
            break
        alpha = _calc_alpha(pack, n)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = pack
            
    # Phase 3: Local Search to improve alpha
    if time.time() < deadline - 5.0 and best_triples:
        best_triples = _local_search_alpha(best_triples, best_alpha, n, deadline)
        best_alpha = _calc_alpha(best_triples, n)

    return best_triples

def _calc_alpha(triples, n):
    if not triples:
        return n
    
    triple_masks = []
    for t in triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
        
    max_size = 0
    # Iterate all subsets
    # Optimization: Check sizes from n down to max_size+1
    # If we find one, we can stop? No, we need max.
    # But if we iterate all, it's 65536.
    # We can prune: if a subset S is independent, all subsets are.
    # We want max size.
    # Just iterate all masks.
    for mask in range(1, 1 << n):
        size = bin(mask).count('1')
        if size <= max_size:
            continue
        is_independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_independent = False
                break
        if is_independent:
            max_size = size
    return max_size

def _local_search_alpha(triples, alpha, n, deadline):
    """
    Local search to minimize alpha.
    Move: Swap a triple with another valid triple.
    """
    current_triples = [t[:] for t in triples]
    current_alpha = alpha
    
    used_pairs = set()
    for t in current_triples:
        for i in range(3):
            for j in range(i+1, 3):
                used_pairs.add(tuple(sorted((t[i], t[j]))))
                
    all_triples = list(itertools.combinations(range(n), 3))
    triple_pairs = []
    for t in all_triples:
        ps = [tuple(sorted((t[0], t[1]))), tuple(sorted((t[0], t[2]))), tuple(sorted((t[1], t[2])))]
        triple_pairs.append(ps)
        
    tri_index = {t: i for i, t in enumerate(all_triples)}
    selected_indices = [tri_index[tuple(t)] for t in current_triples]
    
    no_improve_count = 0
    
    while time.time() < deadline - 5.0 and no_improve_count < 50:
        # Pick a random triple to swap
        idx_remove = random.choice(selected_indices)
        t_remove = all_triples[idx_remove]
        
        # Remove it
        current_triples.remove(list(t_remove))
        selected_indices.remove(idx_remove)
        for p in triple_pairs[idx_remove]:
            used_pairs.remove(p)
            
        # Try to find a replacement that reduces alpha
        # We can sample random valid triples
        improved = False
        # Number of attempts
        attempts = 0
        while attempts < 100:
            idx_add = random.randint(0, len(all_triples) - 1)
            if idx_add == idx_remove:
                attempts += 1
                continue
            t_add = all_triples[idx_add]
            ps = triple_pairs[idx_add]
            
            valid = True
            for p in ps:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                # Add it
                current_triples.append(list(t_add))
                selected_indices.append(idx_add)
                for p in ps:
                    used_pairs.add(p)
                
                new_alpha = _calc_alpha(current_triples, n)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    no_improve_count = 0
                    improved = True
                    break
                else:
                    # Revert
                    current_triples.remove(list(t_add))
                    selected_indices.remove(idx_add)
                    for p in ps:
                        used_pairs.remove(p)
            attempts += 1
            
        if not improved:
            no_improve_count += 1
            # Restore original
            ps_rem = triple_pairs[idx_remove]
            current_triples.append(list(t_remove))
            selected_indices.append(idx_remove)
            for p in ps_rem:
                used_pairs.add(p)

    return current_triples
