import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 40s safety margin
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute bitmasks and pairs for each triple
    triple_masks = [0] * len(all_triples)
    triple_pairs = []
    for i, t in enumerate(all_triples):
        mask = 0
        a, b, c = t
        mask |= (1 << a)
        mask |= (1 << b)
        mask |= (1 << c)
        triple_masks[i] = mask
        triple_pairs.append([(a, b), (a, c), (b, c)])
        
    # Precompute masks for subsets of each size for independence check
    masks_by_size = [[] for _ in range(n + 1)]
    for r in range(1, n + 1):
        for subset in itertools.combinations(points, r):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            masks_by_size[r].append(mask)

    def get_alpha(edge_masks):
        # Check subsets from size n down to 1
        for k in range(n, 0, -1):
            for s_mask in masks_by_size[k]:
                is_independent = True
                for e in edge_masks:
                    if (e & s_mask) == e:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 1

    def try_local_search(triples_list, masks_list, current_alpha, time_limit):
        """
        Perform local search by swapping triples to reduce independence number.
        Returns (improved_triples, improved_masks, improved_alpha) or None.
        """
        t_limit = time_limit
        best_alpha_ls = current_alpha
        best_triples_ls = triples_list.copy()
        best_masks_ls = masks_list.copy()
        
        # Build a mapping from pair to triple index for quick lookup
        pair_to_idx = {}
        for i, t in enumerate(triples_list):
            a, b, c = t
            pair_to_idx[(a, b)] = i
            pair_to_idx[(a, c)] = i
            pair_to_idx[(b, c)] = i
        
        # For each triple in the current solution, try replacing it with another
        # that doesn't conflict with the remaining triples
        remaining_indices = set(range(len(all_triples)))
        for i, t in enumerate(triples_list):
            remaining_indices.discard(triples_list.index(t))
        
        # Actually, let's track which triple indices are used
        used_triple_indices = set()
        for t in triples_list:
            idx = all_triples.index(t)
            used_triple_indices.add(idx)
        
        num_triples = len(triples_list)
        
        for _ in range(200):  # Limit local search iterations
            if time.time() > t_limit:
                break
            
            # Pick a random triple to try to replace
            pos = random.randint(0, num_triples - 1)
            old_triple = triples_list[pos]
            old_mask = masks_list[pos]
            old_idx = all_triples.index(old_triple)
            
            # Remove old triple
            triples_list[pos] = None
            masks_list[pos] = None
            used_triple_indices.discard(old_idx)
            
            # Try to find a replacement
            candidates = [i for i in range(len(all_triples)) 
                         if i not in used_triple_indices]
            random.shuffle(candidates)
            
            found = False
            for cand_idx in candidates:
                cand_triple = all_triples[cand_idx]
                cand_pairs = triple_pairs[cand_idx]
                
                # Check if candidate conflicts with other triples
                conflict = False
                for j in range(num_triples):
                    if j == pos:
                        continue
                    other_triple = triples_list[j]
                    other_mask = masks_list[j]
                    # Check if any pair of cand_triple is in other_triple
                    for p in cand_pairs:
                        if (other_mask & (1 << p[0])) and (other_mask & (1 << p[1])):
                            conflict = True
                            break
                    if conflict:
                        break
                
                if not conflict:
                    triples_list[pos] = cand_triple
                    masks_list[pos] = triple_masks[cand_idx]
                    used_triple_indices.add(cand_idx)
                    found = True
                    break
            
            if not found:
                # Restore old triple
                triples_list[pos] = old_triple
                masks_list[pos] = old_mask
                used_triple_indices.add(old_idx)
                continue
            
            # Evaluate new alpha
            valid_masks = [m for m in masks_list if m is not None]
            new_alpha = get_alpha(valid_masks)
            
            if new_alpha < best_alpha_ls:
                best_alpha_ls = new_alpha
                best_triples_ls = triples_list.copy()
                best_masks_ls = masks_list.copy()
                if best_alpha_ls <= 5:
                    break
        
        if best_alpha_ls < current_alpha:
            return best_triples_ls, best_masks_ls, best_alpha_ls
        return None

    best_alpha = n + 1
    best_triples = []
    
    indices = list(range(len(all_triples)))
    
    # Randomized Greedy Construction with Local Search
    while time.time() < deadline:
        random.shuffle(indices)
        used_pairs = set()
        current_edges = []
        current_masks = []
        
        for t_idx in indices:
            pairs = triple_pairs[t_idx]
            # Check if any pair is already used
            if pairs[0] in used_pairs or pairs[1] in used_pairs or pairs[2] in used_pairs:
                continue
            
            # Add triple
            used_pairs.add(pairs[0])
            used_pairs.add(pairs[1])
            used_pairs.add(pairs[2])
            
            t = all_triples[t_idx]
            current_edges.append(t)
            current_masks.append(triple_masks[t_idx])
            
        # Evaluate independence number
        alpha = get_alpha(current_masks)
        
        # Try local search to improve
        remaining_time = deadline - time.time()
        if remaining_time > 2:  # Only do local search if we have time
            ls_result = try_local_search(current_edges.copy(), current_masks.copy(), 
                                         alpha, time.time() + min(remaining_time - 1, 15))
            if ls_result is not None:
                current_edges, current_masks, alpha = ls_result
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_edges.copy()
            # If we find a very low alpha, it's likely optimal. 
            # Break to save time and avoid timeout.
            if best_alpha <= 5:
                break

    return best_triples
