import random
import time

def construct():
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute structures
    struct_set = set()
    
    for t in all_triples:
        a, b, c = t
        others = [x for x in points if x not in t]
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                for k in range(j+1, len(others)):
                    u, v, w = others[i], others[j], others[k]
                    for perm in [(u, v, w), (u, w, v), (v, u, w), (v, w, u), (w, u, v), (w, v, u)]:
                        u_p, v_p, w_p = perm
                        t1 = tuple(sorted((a, b, c)))
                        t2 = tuple(sorted((a, b, w_p)))
                        t3 = tuple(sorted((a, v_p, c)))
                        t4 = tuple(sorted((u_p, b, c)))
                        t5 = tuple(sorted((a, v_p, w_p)))
                        t6 = tuple(sorted((u_p, b, w_p)))
                        t7 = tuple(sorted((u_p, v_p, c)))
                        t8 = tuple(sorted((u_p, v_p, w_p)))
                        struct_set.add(frozenset([t1, t2, t3, t4, t5, t6, t7, t8]))
                        
    structs = list(struct_set)
    num_structs = len(structs)
    struct_to_idx = {s: i for i, s in enumerate(structs)}
    
    triple_to_structs = [[] for _ in range(num_triples)]
    struct_triples = [[] for _ in range(num_structs)]
    
    for s_idx, s in enumerate(structs):
        for t in s:
            t_idx = triple_to_idx[t]
            triple_to_structs[t_idx].append(s_idx)
            struct_triples[s_idx].append(t_idx)
            
    # Initial Greedy Solution
    def greedy_solve():
        random.shuffle(all_triples)
        current_set = set()
        for t in all_triples:
            t_idx = triple_to_idx[t]
            is_safe = True
            for s_idx in triple_to_structs[t_idx]:
                all_present = True
                for t2_idx in struct_triples[s_idx]:
                    if t2_idx == t_idx: continue
                    if all_triples[t2_idx] not in current_set:
                        all_present = False
                        break
                if all_present:
                    is_safe = False
                    break
            if is_safe:
                current_set.add(t)
        return current_set

    best_set = greedy_solve()
    best_size = len(best_set)
    
    # B&B Setup
    start_time = time.time()
    time_limit = 115
    
    struct_count = [0] * num_structs
    current_triples = set()
    
    def solve(available_indices, current_size):
        nonlocal best_size, best_set
        
        if time.time() - start_time > time_limit:
            return
            
        if current_size + len(available_indices) <= best_size:
            return
            
        if not available_indices:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_triples)
            return
            
        # Heuristic: Pick triple with highest pressure
        best_t_idx = -1
        max_pressure = -1
        for t_idx in available_indices:
            pressure = 0
            for s_idx in triple_to_structs[t_idx]:
                if struct_count[s_idx] > 0:
                    pressure += 1
            if pressure > max_pressure:
                max_pressure = pressure
                best_t_idx = t_idx
                
        t_idx = best_t_idx
        t = all_triples[t_idx]
        
        # Branch 1: Include t
        valid = True
        affected_structs = []
        for s_idx in triple_to_structs[t_idx]:
            if struct_count[s_idx] == 7:
                valid = False
                break
            struct_count[s_idx] += 1
            affected_structs.append(s_idx)
            
        if valid:
            current_triples.add(t_idx)
            new_available = [x for x in available_indices if x != t_idx]
            solve(new_available, current_size + 1)
            current_triples.remove(t_idx)
            
        for s_idx in affected_structs:
            struct_count[s_idx] -= 1
            
        # Branch 2: Exclude t
        new_available = [x for x in available_indices if x != t_idx]
        solve(new_available, current_size)
        
    solve(list(range(num_triples)), 0)
    
    return [list(all_triples[i]) for i in best_set]
