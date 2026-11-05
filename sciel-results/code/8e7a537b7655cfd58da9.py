import time
import random
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute pair sets for each triple index
    triple_pair_sets = []
    for t in all_triples:
        triple_pair_sets.append(set([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]))

    def calc_alpha_milp(triples_indices):
        if not triples_indices:
            return 16
            
        c = [1.0] * 16
        A = []
        for t_idx in triples_indices:
            t = all_triples[t_idx]
            row = [0.0] * 16
            row[t[0]] = 1.0
            row[t[1]] = 1.0
            row[t[2]] = 1.0
            A.append(row)
            
        constraints = LinearConstraint(A, 0, 1)
        integrality = [1] * 16
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            return int(round(res.fun))
        else:
            # Fallback to brute force if MILP fails
            return calc_alpha_brute(triples_indices)

    def calc_alpha_brute(triples_indices):
        if not triples_indices:
            return 16
        masks = []
        for t_idx in triples_indices:
            t = all_triples[t_idx]
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(mask)
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                is_ind = True
                for m in masks:
                    if (m & mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def find_independent_set_milp(triples_indices, target_k):
        if not triples_indices:
            return set(points[:target_k])
            
        c = [1.0] * 16
        A = []
        for t_idx in triples_indices:
            t = all_triples[t_idx]
            row = [0.0] * 16
            row[t[0]] = 1.0
            row[t[1]] = 1.0
            row[t[2]] = 1.0
            A.append(row)
            
        constraints = LinearConstraint(A, 0, 1)
        integrality = [1] * 16
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            max_size = int(round(res.fun))
            if max_size >= target_k:
                S = set()
                for i in range(16):
                    if res.x[i] > 0.5:
                        S.add(i)
                # Return a subset of size target_k
                return set(list(S)[:target_k])
        return None

    def is_compatible(triple_idx, used_pairs):
        for p in triple_pair_sets[triple_idx]:
            if p in used_pairs:
                return False
        return True

    def build_greedy_system():
        indices = list(range(n_triples))
        random.shuffle(indices)
        current = []
        used_pairs = set()
        for idx in indices:
            if is_compatible(idx, used_pairs):
                current.append(idx)
                for p in triple_pair_sets[idx]:
                    used_pairs.add(p)
        return current

    # 1. Initialize
    best_init = []
    max_init_size = 0
    init_deadline = start_time + 5
    while time.time() < init_deadline:
        sys = build_greedy_system()
        if len(sys) > max_init_size:
            max_init_size = len(sys)
            best_init = sys
        if max_init_size >= 38:
            break

    current_triples = best_init
    current_alpha = calc_alpha_milp(current_triples)
    best_alpha = current_alpha
    best_triples = [list(all_triples[i]) for i in current_triples]
    
    used_pairs = set()
    for t_idx in current_triples:
        for p in triple_pair_sets[t_idx]:
            used_pairs.add(p)
            
    no_improve_counter = 0
    
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        current_alpha = calc_alpha_milp(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = [list(all_triples[i]) for i in current_triples]
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            
        target_k = current_alpha
        S = find_independent_set_milp(current_triples, target_k)
        
        if S is None:
            if target_k < best_alpha:
                best_alpha = target_k
                best_triples = [list(all_triples[i]) for i in current_triples]
            continue
        
        # Generate candidate triples inside S
        candidates = []
        for t_idx in range(n_triples):
            t = all_triples[t_idx]
            if t[0] in S and t[1] in S and t[2] in S:
                candidates.append(t_idx)
                
        random.shuffle(candidates)
        
        added = False
        for t_new_idx in candidates:
            if is_compatible(t_new_idx, used_pairs):
                current_triples.append(t_new_idx)
                for p in triple_pair_sets[t_new_idx]:
                    used_pairs.add(p)
                added = True
                break
        
        if not added:
            if current_triples:
                t_old_idx = random.choice(current_triples)
                current_triples.remove(t_old_idx)
                for p in triple_pair_sets[t_old_idx]:
                    used_pairs.discard(p)
                    
                swap_success = False
                for t_new_idx in candidates:
                    if is_compatible(t_new_idx, used_pairs):
                        current_triples.append(t_new_idx)
                        for p in triple_pair_sets[t_new_idx]:
                            used_pairs.add(p)
                        swap_success = True
                        break
                        
                if not swap_success:
                    current_triples.append(t_old_idx)
                    for p in triple_pair_sets[t_old_idx]:
                        used_pairs.add(p)

    return best_triples
