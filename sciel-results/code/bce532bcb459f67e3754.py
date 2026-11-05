import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    
    def is_valid(triple, existing_triples):
        # triple is a tuple
        # existing_triples is a list of tuples
        # Check intersection
        for t in existing_triples:
            # Fast check: 
            # If they share 2 or more points, invalid.
            # We can use set intersection or manual check.
            # Manual check is faster for size 3.
            if (triple[0] in t or triple[1] in t or triple[2] in t):
                # Count matches
                count = 0
                if triple[0] in t: count += 1
                if triple[1] in t: count += 1
                if triple[2] in t: count += 1
                if count > 1:
                    return False
        return True

    def calc_alpha(triples):
        if not triples:
            return N
        t_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            t_masks.append(mask)
        
        best_alpha = 0
        # Iterate all subsets
        for mask in range(1, 1 << N):
            if mask.bit_count() <= best_alpha:
                continue
            
            is_indep = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            
            if is_indep:
                best_alpha = mask.bit_count()
                
        return best_alpha

    # 1. Greedy Maximal Packing
    random.shuffle(all_triples)
    triples = []
    for t in all_triples:
        if is_valid(t, triples):
            triples.append(t)
            
    best_triples = triples[:]
    best_alpha = calc_alpha(triples)
    
    current_triples = best_triples
    current_alpha = best_alpha
    T = 1.0
    
    while time.time() < deadline:
        move_applied = False
        new_triples = current_triples[:]
        
        has_15 = any(15 in t for t in current_triples)
        
        if not has_15:
            # Force integration of 15
            a, b = random.sample(range(15), 2)
            candidate = (15, a, b)
            if is_valid(candidate, current_triples):
                new_triples.append(candidate)
                move_applied = True
            else:
                # Find conflicting triple
                target_t = None
                for t in current_triples:
                    if a in t and b in t:
                        target_t = t
                        break
                
                if target_t is not None:
                    new_triples.remove(target_t)
                    new_triples.append(candidate)
                    move_applied = True
        else:
            move_type = random.choice(['swap', 'remove', 'add'])
            if move_type == 'swap' and current_triples:
                idx = random.randint(0, len(current_triples) - 1)
                removed = current_triples[idx]
                new_triples.pop(idx)
                candidate = random.choice(all_triples)
                if candidate not in new_triples and is_valid(candidate, new_triples):
                    new_triples.append(candidate)
                    move_applied = True
            elif move_type == 'remove' and current_triples:
                idx = random.randint(0, len(current_triples) - 1)
                new_triples.pop(idx)
                move_applied = True
            elif move_type == 'add':
                candidate = random.choice(all_triples)
                if candidate not in current_triples and is_valid(candidate, current_triples):
                    new_triples.append(candidate)
                    move_applied = True
        
        if not move_applied:
            continue
            
        new_alpha = calc_alpha(new_triples)
        
        if new_alpha < current_alpha:
            current_triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
            T = max(T * 0.95, 0.1)
        else:
            delta = new_alpha - current_alpha
            if delta > 0:
                prob = pow(2.71828, -delta / T)
                if random.random() < prob:
                    current_triples = new_triples
                    current_alpha = new_alpha
            T = max(T * 0.99, 0.1)

    # Convert tuples to lists for output
    return [list(t) for t in best_triples]
