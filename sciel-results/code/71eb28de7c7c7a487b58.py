import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s

    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))

    # Precompute pair for each triple (3 pairs per triple)
    triple_pairs = {}
    for t in all_triples:
        triple_pairs[t] = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]

    # Precompute triple bitmask
    triple_mask = {}
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_mask[t] = m

    def compute_alpha(triples):
        """Compute independence number using bitmask enumeration with pruning."""
        if not triples:
            return n
        masks = [triple_mask[t] for t in triples]
        max_alpha = 0
        # Enumerate all 2^16 subsets
        # Optimization: if a subset contains a triple, skip all supersets? 
        # Actually we just need max, so enumerate all and track max.
        # 65536 * len(triples) checks. With ~40 triples that's ~2.6M ops.
        # Let's do it efficiently.
        for mask in range(1, 1 << n):
            # Quick check: does this mask contain any triple?
            independent = True
            for tm in masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                sz = mask.bit_count()
                if sz > max_alpha:
                    max_alpha = sz
                    if max_alpha == n:
                        return n
        return max_alpha

    def compute_alpha_fast(triples):
        """Faster: use the fact that we want max independent set.
        Use a different approach: for each subset size from n down, check if any independent set of that size exists.
        But that's also expensive. Let's stick with the enumeration but optimize the inner loop.
        """
        if not triples:
            return n
        masks = [triple_mask[t] for t in triples]
        max_alpha = 0
        for mask in range(1, 1 << n):
            independent = True
            for tm in masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                sz = mask.bit_count()
                if sz > max_alpha:
                    max_alpha = sz
        return max_alpha

    def greedy_construct():
        """Greedy: at each step, pick the triple that most reduces alpha."""
        used_pairs = set()
        triples = []
        
        # First pass: just greedily add triples to get a dense system
        # Then refine with alpha-based selection
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            pairs = triple_pairs[t]
            if all(p not in used_pairs for p in pairs):
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        
        return triples, used_pairs

    def greedy_alpha_construct():
        """Greedy: start empty, at each step pick the triple that most reduces alpha.
        This is expensive but more targeted."""
        used_pairs = set()
        triples = []
        current_alpha = n
        
        # Precompute for each triple, which other triples conflict with it
        # Two triples conflict if they share a pair
        conflict_map = {t: set() for t in all_triples}
        for i, t1 in enumerate(all_triples):
            for t2 in all_triples[i+1:]:
                # Check if they share a pair
                s1 = set(t1)
                s2 = set(t2)
                common = s1 & s2
                if len(common) >= 2:
                    conflict_map[t1].add(t2)
                    conflict_map[t2].add(t1)
        
        available = set(all_triples)
        
        while available:
            if time.time() > deadline:
                break
            
            best_t = None
            best_alpha = current_alpha
            best_reduction = 0
            
            # Sample a subset of available triples for efficiency
            if len(available) > 200:
                candidates = random.sample(list(available), 200)
            else:
                candidates = list(available)
            
            for t in candidates:
                # Check if t is valid (no pair conflict with existing)
                pairs = triple_pairs[t]
                if any(p in used_pairs for p in pairs):
                    continue
                
                # Tentatively add t and compute new alpha
                new_triples = triples + [t]
                new_alpha = compute_alpha(new_triples)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_t = t
                    best_reduction = current_alpha - new_alpha
                    if best_alpha <= 4:
                        break
            
            if best_t is None:
                # No valid triple improves alpha; break
                break
            
            triples.append(best_t)
            for p in triple_pairs[best_t]:
                used_pairs.add(p)
            available.discard(best_t)
            # Remove conflicting triples from available
            for conf in conflict_map[best_t]:
                available.discard(conf)
            
            current_alpha = best_alpha
            if current_alpha <= 4:
                break
        
        return triples, used_pairs

    def local_search(triples, used_pairs):
        """Local search: try to reduce alpha by swapping triples."""
        if time.time() > deadline:
            return triples, used_pairs, compute_alpha(triples)
        
        current_alpha = compute_alpha(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        best_used = set(used_pairs)
        
        # Precompute conflict map
        conflict_map = {}
        for t in all_triples:
            conflict_map[t] = set()
        for i, t1 in enumerate(all_triples):
            for t2 in all_triples[i+1:]:
                if len(set(t1) & set(t2)) >= 2:
                    conflict_map[t1].add(t2)
                    conflict_map[t2].add(t1)
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve < max_no_improve:
            if not triples:
                break
            
            # Pick a random triple to remove
            idx = random.randint(0, len(triples) - 1)
            remove_t = triples[idx]
            
            # Remove it
            triples.pop(idx)
            for p in triple_pairs[remove_t]:
                used_pairs.discard(p)
            
            # Find all valid triples that can be added
            valid_adds = []
            for t in all_triples:
                pairs = triple_pairs[t]
                if all(p not in used_pairs for p in pairs):
                    valid_adds.append(t)
            
            if not valid_adds:
                # Put back
                triples.insert(idx, remove_t)
                for p in triple_pairs[remove_t]:
                    used_pairs.add(p)
                no_improve += 1
                continue
            
            # Try a random valid add
            add_t = random.choice(valid_adds)
            triples.append(add_t)
            for p in triple_pairs[add_t]:
                used_pairs.add(p)
            
            new_alpha = compute_alpha(triples)
            
            if new_alpha <= current_alpha:
                current_alpha = new_alpha
                no_improve = 0
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = triples[:]
                    best_used = set(used_pairs)
                    if best_alpha <= 4:
                        return best_triples, best_used, best_alpha
            else:
                # Revert
                triples.pop()
                for p in triple_pairs[add_t]:
                    used_pairs.discard(p)
                triples.insert(idx, remove_t)
                for p in triple_pairs[remove_t]:
                    used_pairs.add(p)
                no_improve += 1
        
        return best_triples, best_used, best_alpha

    def multi_start():
        """Multiple random restarts with greedy + local search."""
        global_best = None
        global_best_alpha = n + 1
        
        for restart in range(30):
            if time.time() > deadline:
                break
            
            # Random greedy packing
            triples, used_pairs = greedy_construct()
            alpha = compute_alpha(triples)
            
            if alpha < global_best_alpha:
                global_best_alpha = alpha
                global_best = (triples[:], set(used_pairs))
                if global_best_alpha <= 4:
                    return global_best[0]
            
            # Local search
            if time.time() < deadline - 10:
                triples2, used2, alpha2 = local_search(triples, used_pairs)
                if alpha2 < global_best_alpha:
                    global_best_alpha = alpha2
                    global_best = (triples2[:], set(used2))
                    if global_best_alpha <= 4:
                        return global_best[0]
        
        if global_best is not None:
            return global_best[0]
        return []

    # Run multi-start
    result = multi_start()
    return [list(t) for t in result]
