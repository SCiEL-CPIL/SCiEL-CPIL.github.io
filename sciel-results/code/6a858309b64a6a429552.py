import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    points = list(range(n))
    
    masks_by_size = [[] for _ in range(n + 1)]
    for k in range(n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            masks_by_size[k].append(mask)
            
    triples = []
    pair_used = {}
    triple_masks = []
    
    def add_triple(a, b, c):
        t = tuple(sorted((a, b, c)))
        if t in triples:
            return False
        p1, p2, p3 = (t[0], t[1]), (t[1], t[2]), (t[0], t[2])
        if p1 in pair_used or p2 in pair_used or p3 in pair_used:
            return False
        triples.append(t)
        pair_used[p1] = True
        pair_used[p2] = True
        pair_used[p3] = True
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        return True

    def remove_triple(t):
        if t not in triples:
            return False
        idx = triples.index(t)
        triples.pop(idx)
        a, b, c = t
        pair_used.pop((a, b), None)
        pair_used.pop((b, c), None)
        pair_used.pop((a, c), None)
        mask = (1 << a) | (1 << b) | (1 << c)
        triple_masks.pop(idx)
        return True

    def get_random_valid_triple():
        shuffled = points[:]
        random.shuffle(shuffled)
        for i in range(len(shuffled)):
            a = shuffled[i]
            for j in range(i+1, len(shuffled)):
                b = shuffled[j]
                if (a, b) in pair_used:
                    continue
                for k in range(j+1, len(shuffled)):
                    c = shuffled[k]
                    if (a, c) in pair_used or (b, c) in pair_used:
                        continue
                    return (a, b, c)
        return None

    def is_independent(mask, t_masks):
        for tm in t_masks:
            if (mask & tm) == tm:
                return False
        return True

    def calc_alpha(t_masks, start_k=None):
        if not t_masks:
            return n
        if start_k is None:
            start_k = n
        # We want max k such that exists independent set of size k
        # We check from start_k down to 1
        for k in range(start_k, 0, -1):
            for mask in masks_by_size[k]:
                if is_independent(mask, t_masks):
                    return k
        return 0

    # Initial Construction
    attempts = 0
    while attempts < 5000:
        t = get_random_valid_triple()
        if t is None:
            break
        add_triple(*t)
        attempts += 1

    current_alpha = calc_alpha(triple_masks)
    best_alpha = current_alpha
    best_triples = triples.copy()
    best_triple_masks = triple_masks.copy()
    
    T = 1.0
    cooling = 0.995
    
    while time.time() < deadline:
        move_type = random.random()
        
        # Store state
        old_triples = triples
        old_pair_used = pair_used
        old_triple_masks = triple_masks
        
        new_triples = old_triples.copy()
        new_pair_used = old_pair_used.copy()
        new_triple_masks = old_triple_masks.copy()
        
        added = False
        removed = False
        
        if move_type < 0.4:
            t = get_random_valid_triple()
            if t is None:
                continue
            a, b, c = t
            if (a,b) in new_pair_used or (b,c) in new_pair_used or (a,c) in new_pair_used:
                continue
            new_triples.append(t)
            new_pair_used[(a,b)] = True
            new_pair_used[(b,c)] = True
            new_pair_used[(a,c)] = True
            mask = (1 << a) | (1 << b) | (1 << c)
            new_triple_masks.append(mask)
            added = True
        elif move_type < 0.7:
            if not new_triples:
                continue
            idx = random.randint(0, len(new_triples)-1)
            t = new_triples[idx]
            a, b, c = t
            new_triples.pop(idx)
            new_pair_used.pop((a,b), None)
            new_pair_used.pop((b,c), None)
            new_pair_used.pop((a,c), None)
            new_triple_masks.pop(idx)
            removed = True
        else:
            if not new_triples:
                continue
            idx = random.randint(0, len(new_triples)-1)
            t_old = new_triples[idx]
            a, b, c = t_old
            new_triples.pop(idx)
            new_pair_used.pop((a,b), None)
            new_pair_used.pop((b,c), None)
            new_pair_used.pop((a,c), None)
            new_triple_masks.pop(idx)
            removed = True
            
            t_new = get_random_valid_triple()
            if t_new is None:
                continue
            x, y, z = t_new
            if (x,y) in new_pair_used or (y,z) in new_pair_used or (x,z) in new_pair_used:
                continue
            new_triples.append(t_new)
            new_pair_used[(x,y)] = True
            new_pair_used[(y,z)] = True
            new_pair_used[(x,z)] = True
            mask = (1 << x) | (1 << y) | (1 << z)
            new_triple_masks.append(mask)
            added = True

        # Calculate new alpha
        if added and not removed:
            # Alpha can only decrease or stay same
            # Check if current_alpha is still achievable
            # If yes, new_alpha = current_alpha
            # If no, find new alpha
            possible = True
            for mask in masks_by_size[current_alpha]:
                if is_independent(mask, new_triple_masks):
                    possible = True
                    break
            else:
                possible = False
            
            if possible:
                new_alpha = current_alpha
            else:
                new_alpha = calc_alpha(new_triple_masks, start_k=current_alpha - 1)
                
        elif removed and not added:
            # Alpha can only increase or stay same
            # We need to find the new max
            # It is at least current_alpha
            # Check from n down to current_alpha
            new_alpha = calc_alpha(new_triple_masks, start_k=n)
            
        else:
            # Swap or mixed
            new_alpha = calc_alpha(new_triple_masks, start_k=n)

        delta = new_alpha - current_alpha
        
        if delta <= 0:
            accept = True
        else:
            if T > 0.1:
                accept = random.random() < (2.71828 ** (-delta / T))
            else:
                accept = False
        
        if accept:
            triples = new_triples
            pair_used = new_pair_used
            triple_masks = new_triple_masks
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = triples.copy()
                best_triple_masks = triple_masks.copy()
            
        T *= cooling
        
    return [list(t) for t in best_triples]
