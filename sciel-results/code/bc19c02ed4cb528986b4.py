"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = 145.0

    def num_to_vec(num, length=6):
        vec = [0] * length
        for i in range(length):
            vec[i] = num % 3
            num //= 3
        return vec

    def vec_to_num(vec):
        num = 0
        for i in range(len(vec)):
            num += vec[i] * (3 ** i)
        return num

    # Precompute cover sets
    cover_set = []
    for i in range(729):
        v = num_to_vec(i)
        s = [i]
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    s.append(vec_to_num(v))
            v[pos] = orig
        cover_set.append(s)

    # coverers[w] = list of codewords that cover word w
    coverers = [[] for _ in range(729)]
    for c in range(729):
        for w in cover_set[c]:
            coverers[w].append(c)

    def greedy_cover():
        uncovered = np.ones(729, dtype=bool)
        selected = []
        while np.any(uncovered):
            best_c = -1
            max_gain = 0
            for c in range(729):
                if c in selected:
                    continue
                gain = np.sum(uncovered[cover_set[c]])
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            if max_gain == 0:
                break
            selected.append(best_c)
            uncovered[cover_set[best_c]] = False
        return set(selected)

    def simulated_annealing(initial_set, max_time=30.0):
        current_set = set(initial_set)
        best_set = set(initial_set)
        best_size = len(current_set)
        
        cover_count = np.zeros(729, dtype=np.int32)
        for c in current_set:
            for w in cover_set[c]:
                cover_count[w] += 1
        
        start_sa = time.time()
        T = 1.0
        cooling = 0.9995
        
        while time.time() - start_sa < max_time and T > 0.01:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4 and len(current_set) > 75:
                # Try deletion
                candidates = list(current_set)
                random.shuffle(candidates)
                for c in candidates[:50]:
                    can_remove = True
                    for w in cover_set[c]:
                        if cover_count[w] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Perform deletion
                        current_set.remove(c)
                        for w in cover_set[c]:
                            cover_count[w] -= 1
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break
            elif move_type < 0.8:
                # Try swap: remove c_out, add c_in
                c_out_candidates = list(current_set)
                random.shuffle(c_out_candidates)
                
                for c_out in c_out_candidates[:30]:
                    # Find critical words
                    critical_words = [w for w in cover_set[c_out] if cover_count[w] == 1]
                    if not critical_words:
                        continue
                    
                    # Find candidate c_in that covers all critical words
                    candidate_c_ins = set(coverers[critical_words[0]])
                    for w in critical_words[1:]:
                        candidate_c_ins &= set(coverers[w])
                    candidate_c_ins -= current_set
                    candidate_c_ins.discard(c_out)
                    
                    if not candidate_c_ins:
                        continue
                    
                    c_in_list = list(candidate_c_ins)
                    random.shuffle(c_in_list)
                    
                    for c_in in c_in_list[:15]:
                        # Perform swap
                        current_set.remove(c_out)
                        current_set.add(c_in)
                        
                        for w in cover_set[c_out]:
                            cover_count[w] -= 1
                        for w in cover_set[c_in]:
                            cover_count[w] += 1
                        
                        # Try to delete
                        del_found = False
                        del_candidates = list(current_set)
                        random.shuffle(del_candidates)
                        for c_del in del_candidates[:50]:
                            can_remove = True
                            for w in cover_set[c_del]:
                                if cover_count[w] == 1:
                                    can_remove = False
                                    break
                            if can_remove:
                                current_set.remove(c_del)
                                for w in cover_set[c_del]:
                                    cover_count[w] -= 1
                                if len(current_set) < best_size:
                                    best_size = len(current_set)
                                    best_set = set(current_set)
                                del_found = True
                                break
                        
                        if not del_found:
                            # Accept with probability based on temperature
                            # Here we always revert if no improvement, but in SA we might accept worse
                            # For now, revert
                            current_set.remove(c_in)
                            current_set.add(c_out)
                            for w in cover_set[c_in]:
                                cover_count[w] -= 1
                            for w in cover_set[c_out]:
                                cover_count[w] += 1
                        
                        break
                    break
            else:
                # Random perturbation: remove 2, add 2
                if len(current_set) > 78:
                    remove_list = random.sample(list(current_set), 2)
                    for c in remove_list:
                        current_set.remove(c)
                        for w in cover_set[c]:
                            cover_count[w] -= 1
                    
                    # Add 2 random codewords that help
                    uncovered_words = [w for w in range(729) if cover_count[w] == 0]
                    
                    if uncovered_words:
                        added = 0
                        for w in uncovered_words[:5]:
                            candidates = [c for c in coverers[w] if c not in current_set]
                            if candidates:
                                c_add = random.choice(candidates)
                                current_set.add(c_add)
                                for ww in cover_set[c_add]:
                                    cover_count[ww] += 1
                                added += 1
                                if added == 2:
                                    break
                    else:
                        all_candidates = [c for c in range(729) if c not in current_set]
                        if len(all_candidates) >= 2:
                            for c_add in random.sample(all_candidates, 2):
                                current_set.add(c_add)
                                for ww in cover_set[c_add]:
                                    cover_count[ww] += 1
            
            T *= cooling
        
        return best_set

    # Run greedy + simulated annealing multiple times
    best_overall = None
    best_overall_size = float('inf')
    
    for trial in range(8):
        if time.time() > deadline - 15:
            break
        
        random.seed(trial * 12345 + 67890)
        initial = greedy_cover()
        result = simulated_annealing(initial, max_time=18.0)
        
        if len(result) < best_overall_size:
            best_overall_size = len(result)
            best_overall = result
        
        if best_overall_size <= 73:
            break

    if best_overall is not None:
        # Verify
        cc = np.zeros(729, dtype=np.int32)
        for c in best_overall:
            for w in cover_set[c]:
                cc[w] += 1
        if np.min(cc) >= 1:
            return [num_to_vec(c) for c in best_overall]

    # Fallback
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

# EVOLVE-BLOCK-END
