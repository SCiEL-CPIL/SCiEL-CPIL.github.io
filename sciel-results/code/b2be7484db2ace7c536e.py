import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute pair sets for each triple
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        p1 = (min(a,b), max(a,b))
        p2 = (min(a,c), max(a,c))
        p3 = (min(b,c), max(b,c))
        triple_pairs[t] = frozenset({p1, p2, p3})
    
    # Precompute compatibility: which triples are compatible with each other
    # Two triples are compatible if they share at most one point (no common pair)
    compatible_with = {t: set() for t in all_triples}
    for i, t1 in enumerate(all_triples):
        for t2 in all_triples[i+1:]:
            if not triple_pairs[t1] & triple_pairs[t2]:
                compatible_with[t1].add(t2)
                compatible_with[t2].add(t1)
    
    def calc_alpha(triple_list):
        """Calculate independence number efficiently using bit masks"""
        if not triple_list:
            return n
        
        # Build a bitmask of all points that are "covered" by at least one triple
        # Actually, we need to find the largest set with NO triple contained in it
        # A set S is independent if for every triple t, t is not a subset of S
        
        # Use bitmasks: represent each triple as a 16-bit mask
        triple_masks = []
        for t in triple_list:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
        
        # For each subset of points (represented as bitmask), check if it's independent
        # We want the largest independent set
        # Iterate from largest to smallest
        # Optimization: if a set of size k is not independent, no superset is independent
        # So we can use a BFS/DFS approach
        
        # Better approach: iterate all 2^16 = 65536 possible subsets
        # For each, check if it contains any triple
        # Precompute for each triple mask, which subsets contain it?
        
        # Simple approach: for each subset mask, check all triple masks
        # 65536 * len(triple_list) operations
        # If len(triple_list) ~ 30, that's ~2M operations, which is fine
        
        best = 0
        for subset_mask in range(1 << n):
            # Count bits in subset_mask
            size = bin(subset_mask).count('1')
            if size <= best:
                continue
            
            # Check if subset_mask contains any triple
            independent = True
            for tm in triple_masks:
                if (subset_mask & tm) == tm:
                    independent = False
                    break
            
            if independent:
                best = size
                if best == n:
                    break
        
        return best

    best_alpha = n
    best_triples = []
    
    # Strategy 1: Greedy construction with randomization
    for iteration in range(2000):
        if time.time() > deadline - 30:  # Leave time for local search
            break
        
        # Randomize the order
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        current_triples = []
        used_pairs = set()
        
        for t in shuffled:
            if used_pairs.isdisjoint(triple_pairs[t]):
                current_triples.append(t)
                used_pairs.update(triple_pairs[t])
        
        if not current_triples:
            continue
        
        alpha = calc_alpha(current_triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples.copy()
            
            if best_alpha <= 6:
                # Good enough, break early and focus on local search
                break
    
    # Strategy 2: Local search to improve
    if best_triples:
        current_set = set(best_triples)
        current_alpha = best_alpha
        no_improve_count = 0
        
        for ls_iter in range(500):
            if time.time() > deadline - 5:
                break
            
            if no_improve_count > 50:
                break
            
            # Try multiple swap candidates
            found_improvement = False
            triples_list = list(current_set)
            random.shuffle(triples_list)
            
            for t_out in triples_list:
                if time.time() > deadline - 5:
                    break
                
                new_set = current_set - {t_out}
                
                # Find compatible triples to add
                candidates = [t for t in all_triples 
                             if t not in new_set and 
                             not (triple_pairs[t] & set().union(*[triple_pairs[x] for x in new_set]) if new_set else set())]
                
                # Better: check compatibility properly
                candidates = []
                for t_in in all_triples:
                    if t_in in new_set:
                        continue
                    # Check if t_in is compatible with all triples in new_set
                    compatible = True
                    for t_exist in new_set:
                        if triple_pairs[t_in] & triple_pairs[t_exist]:
                            compatible = False
                            break
                    if compatible:
                        candidates.append(t_in)
                
                if not candidates:
                    continue
                
                # Try a few random candidates
                random.shuffle(candidates)
                for t_in in candidates[:5]:
                    candidate_set = new_set | {t_in}
                    alpha_new = calc_alpha(list(candidate_set))
                    
                    if alpha_new < current_alpha:
                        current_set = candidate_set
                        current_alpha = alpha_new
                        best_alpha = current_alpha
                        best_triples = list(current_set)
                        no_improve_count = 0
                        found_improvement = True
                        break
                    elif alpha_new == current_alpha:
                        # Accept same alpha to explore
                        if random.random() < 0.3:
                            current_set = candidate_set
                            best_triples = list(current_set)
                            break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                no_improve_count += 1
    
    return best_triples
