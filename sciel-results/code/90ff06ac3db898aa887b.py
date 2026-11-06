import itertools
import random
import time
import math

def construct():
    deadline = time.time() + 145
    n = 16
    rng = random.Random(42)
    
    # Precompute sorted masks by popcount descending
    all_masks = list(range(1, 1 << n))
    masks_sorted = sorted(all_masks, key=lambda x: x.bit_count(), reverse=True)
    
    # Precompute all triples
    all_triples_info = []
    for t in itertools.combinations(range(n), 3):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        pairs = frozenset({(t[0], t[1]), (t[0], t[2]), (t[1], t[2])})
        all_triples_info.append((mask, pairs, t))
        
    triple_pair_map = {t: pairs for mask, pairs, t in all_triples_info}
    triple_mask_map = {t: mask for mask, pairs, t in all_triples_info}
    all_triple_tuples = [t for mask, pairs, t in all_triples_info]
    
    def calc_alpha(triple_masks):
        if not triple_masks:
            return n
        for M in masks_sorted:
            is_indep = True
            for T in triple_masks:
                if (T & M) == T:
                    is_indep = False
                    break
            if is_indep:
                return M.bit_count()
        return 0

    def init_system():
        system = set()
        used_pairs = set()
        shuffled = all_triples_info[:]
        rng.shuffle(shuffled)
        for mask, pairs, t in shuffled:
            if not (used_pairs & pairs):
                system.add(t)
                used_pairs |= pairs
        masks = [triple_mask_map[t] for t in system]
        return system, used_pairs, masks

    system, used_pairs, current_masks = init_system()
    best_system = set(system)
    best_masks = list(current_masks)
    best_alpha = calc_alpha(best_masks)
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    start_time = time.time()
    
    while time.time() < deadline - 5:
        if T < min_T:
            break
            
        current_alpha = calc_alpha(current_masks)
        
        move_type = rng.random()
        
        new_system = set(system)
        new_used_pairs = set(used_pairs)
        
        if move_type < 0.5:
            # Swap
            if not system:
                continue
            t_remove = rng.choice(list(system))
            p_remove = triple_pair_map[t_remove]
            new_system.remove(t_remove)
            new_used_pairs -= p_remove
            
            added = False
            for _ in range(50):
                t_add = rng.choice(all_triple_tuples)
                if t_add in new_system:
                    continue
                p_add = triple_pair_map[t_add]
                if not (new_used_pairs & p_add):
                    new_system.add(t_add)
                    new_used_pairs |= p_add
                    added = True
                    break
            
            if not added:
                continue
                
        elif move_type < 0.7:
            # Remove
            if not system:
                continue
            t_remove = rng.choice(list(system))
            p_remove = triple_pair_map[t_remove]
            new_system.remove(t_remove)
            new_used_pairs -= p_remove
            
        elif move_type < 0.9:
            # Add
            added = False
            for _ in range(50):
                t_add = rng.choice(all_triple_tuples)
                if t_add in system:
                    continue
                p_add = triple_pair_map[t_add]
                if not (used_pairs & p_add):
                    new_system.add(t_add)
                    new_used_pairs |= p_add
                    added = True
                    break
            if not added:
                continue
        else:
            # Perturbation
            num_remove = rng.randint(3, 5)
            for _ in range(num_remove):
                if not system:
                    break
                t_remove = rng.choice(list(system))
                p_remove = triple_pair_map[t_remove]
                new_system.remove(t_remove)
                new_used_pairs -= p_remove
            
            for _ in range(num_remove + 2):
                for _ in range(20):
                    t_add = rng.choice(all_triple_tuples)
                    if t_add in new_system:
                        continue
                    p_add = triple_pair_map[t_add]
                    if not (new_used_pairs & p_add):
                        new_system.add(t_add)
                        new_used_pairs |= p_add
                        break

        new_masks = [triple_mask_map[t] for t in new_system]
        new_alpha = calc_alpha(new_masks)
        
        if new_alpha < current_alpha:
            system = new_system
            used_pairs = new_used_pairs
            current_masks = new_masks
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_system = set(system)
                best_masks = list(current_masks)
        else:
            delta = new_alpha - current_alpha
            if delta == 0:
                if rng.random() < 0.5:
                    system = new_system
                    used_pairs = new_used_pairs
                    current_masks = new_masks
            else:
                try:
                    prob = math.exp(-delta / T)
                except OverflowError:
                    prob = 0.0
                if rng.random() < prob:
                    system = new_system
                    used_pairs = new_used_pairs
                    current_masks = new_masks
        
        T *= cooling_rate

    return [list(t) for t in best_system]
