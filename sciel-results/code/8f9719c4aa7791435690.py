import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    points = list(range(16))
    
    # Precompute all triples as tuples and their bitmasks
    all_triples = list(itertools.combinations(points, 3))
    triple_masks = [0] * len(all_triples)
    for i, t in enumerate(all_triples):
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[i] = mask
    
    # Helper to check if a triple can be added (linear constraint)
    def can_add(triples, new_t):
        t_set = set(new_t)
        for t in triples:
            if len(t_set & set(t)) > 1:
                return False
        return True
    
    # Fast alpha calculation using bitmask
    # A subset S is independent if no triple is fully contained in S
    # We want the maximum size of such S
    def calc_alpha_fast(triples):
        if not triples:
            return 16
        
        # For each point, track which triples contain it
        # But simpler: iterate subsets and check if any triple is subset
        
        # Optimize: precompute for each triple its mask
        triple_mask_list = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_mask_list.append(mask)
        
        max_alpha = 0
        # Iterate all subsets
        for mask in range(1 << 16):
            # Check if any triple is subset of mask
            is_independent = True
            for tm in triple_mask_list:
                # Check if tm is subset of mask: (mask & tm) == tm
                if (mask & tm) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                size = mask.bit_count() if hasattr(mask, 'bit_count') else bin(mask).count('1')
                if size > max_alpha:
                    max_alpha = size
        
        return max_alpha
    
    # Greedy strategy: add triples that most reduce alpha
    def greedy_build():
        triples = []
        current_alpha = 16
        
        # Keep trying to add triples that reduce alpha
        while time.time() < deadline - 5:  # Leave 5s margin
            # Find all addable triples
            addable = []
            for t in all_triples:
                if can_add(triples, t):
                    addable.append(t)
            
            if not addable:
                break
            
            # Try each addable triple, pick the one that most reduces alpha
            best_t = None
            best_alpha = current_alpha
            best_triples = triples
            
            # Sample if too many candidates
            if len(addable) > 50:
                random.shuffle(addable)
                addable = addable[:50]
            
            for t in addable:
                new_triples = triples + [t]
                a = calc_alpha_fast(new_triples)
                if a < best_alpha:
                    best_alpha = a
                    best_t = t
                    best_triples = new_triples
            
            if best_t is None:
                # No improvement possible, stop
                break
            
            triples = best_triples
            current_alpha = best_alpha
        
        return triples, current_alpha
    
    # Local search: try swaps to improve
    def local_search(triples, current_alpha):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Try removing one triple and adding another
            for i in range(len(triples)):
                if time.time() > deadline - 3:
                    break
                
                # Remove triple i
                remaining = triples[:i] + triples[i+1:]
                
                # Find addable triples
                addable = []
                for t in all_triples:
                    if t not in remaining and can_add(remaining, t):
                        addable.append(t)
                
                if not addable:
                    continue
                
                # Try a few candidates
                random.shuffle(addable)
                for t in addable[:20]:
                    new_triples = remaining + [t]
                    a = calc_alpha_fast(new_triples)
                    if a < current_alpha:
                        triples = new_triples
                        current_alpha = a
                        improved = True
                        break
                
                if improved:
                    break
        
        return triples, current_alpha
    
    # Main search loop
    best_alpha = 16
    best_triples = []
    
    # Multiple restarts with different strategies
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # Strategy 1: Pure greedy
        if num_restarts % 2 == 1:
            init_triples, alpha = greedy_build()
        else:
            # Strategy 2: Random initial + local search
            init_triples = []
            random.shuffle(all_triples)
            for t in all_triples:
                if can_add(init_triples, t):
                    init_triples.append(t)
            alpha = calc_alpha_fast(init_triples)
        
        # Local search polish
        improved_triples, final_alpha = local_search(init_triples, alpha)
        
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_triples = improved_triples
            
            # If we reach a good value, we can try to break early
            if best_alpha <= 6:
                # Continue searching for a bit more
                if time.time() > deadline - 20:
                    break
    
    # Final polish with more time
    if best_triples and time.time() < deadline - 5:
        final_triples, final_alpha = local_search(best_triples, best_alpha)
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_triples = final_triples
    
    # If best_triples is empty (should not happen), return seed
    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return [list(t) for t in best_triples]

# EVOLVE-BLOCK-END
