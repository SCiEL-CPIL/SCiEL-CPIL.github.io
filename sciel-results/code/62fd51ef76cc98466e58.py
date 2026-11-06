import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2)s
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    # Generate all 6-subsets
    for six_subset in combinations(range(n), 6):
        v = list(six_subset)
        for i in range(1, 6):
            p1 = (v[0], v[i])
            rem = [v[j] for j in range(6) if j != 0 and j != i]
            for j in range(1, 4):
                p2 = (rem[0], rem[j])
                if j == 1:
                    p3 = (rem[2], rem[3])
                elif j == 2:
                    p3 = (rem[1], rem[3])
                else:
                    p3 = (rem[1], rem[2])
                
                t_ids = []
                for x in p1:
                    for y in p2:
                        for z in p3:
                            t = tuple(sorted((x, y, z)))
                            t_id = triple_to_id[t]
                            t_ids.append(t_id)
                
                k_idx = len(k222_triples)
                k222_triples.append(t_ids)
                
                for t_id in t_ids:
                    triple_to_k222s[t_id].append(k_idx)
                    
    num_k222s = len(k222_triples)
    
    # Precompute which Ks contain each triple for faster swap checks
    triple_k_set = [set(triple_to_k222s[t]) for t in range(num_triples)]
    
    def solve_instance(seed):
        rng = random.Random(seed)
        active = [False] * num_triples
        k_count = [0] * num_k222s
        
        # Greedy Initialization with constraint-aware ordering
        # Sort triples by their number of K constraints (fewer = easier to add)
        order = list(range(num_triples))
        order.sort(key=lambda t: len(triple_to_k222s[t]))
        rng.shuffle(order)
        
        for t_id in order:
            ok = True
            for k_idx in triple_to_k222s[t_id]:
                if k_count[k_idx] >= 7:
                    ok = False
                    break
            if ok:
                active[t_id] = True
                for k_idx in triple_to_k222s[t_id]:
                    k_count[k_idx] += 1
        
        current_size = sum(active)
        best_size = current_size
        best_active = active[:]
        
        # Simulated annealing with swap moves
        steps = 8000
        T0 = 1.0
        T_final = 0.01
        cooling = (T_final / T0) ** (1.0 / max(steps, 1))
        T = T0
        
        # Track recent best to restart if stuck
        last_improve = 0
        
        for step in range(steps):
            if time.time() > DEADLINE:
                break
            
            T *= cooling
            
            # Try to add a random inactive triple
            t_in = rng.randint(0, num_triples - 1)
            if not active[t_in]:
                can_add = True
                for k_idx in triple_to_k222s[t_in]:
                    if k_count[k_idx] >= 7:
                        can_add = False
                        break
                if can_add:
                    active[t_in] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_in]:
                        k_count[k_idx] += 1
                    last_improve = step
                    if current_size > best_size:
                        best_size = current_size
                        best_active = active[:]
                    continue
            
            # If cannot add, try a swap
            t_out = rng.randint(0, num_triples - 1)
            if not active[t_out]:
                continue
                
            t_in = rng.randint(0, num_triples - 1)
            if active[t_in]:
                continue
            
            # Check if swap is valid using set intersection for speed
            valid_swap = True
            for k_idx in triple_to_k222s[t_in]:
                t_out_in_k = t_out in k222_triples[k_idx]
                new_count = k_count[k_idx] - (1 if t_out_in_k else 0) + 1
                if new_count >= 8:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Execute swap
                active[t_out] = False
                for k_idx in triple_to_k222s[t_out]:
                    k_count[k_idx] -= 1
                
                active[t_in] = True
                for k_idx in triple_to_k222s[t_in]:
                    k_count[k_idx] += 1
                # current_size unchanged
                
                # With some probability, accept even if we're in a local optimum
                # (this is already handled since swaps maintain size)
                # The key is that swaps can lead to new addable triples
                if step - last_improve > 200 and rng.random() < 0.1:
                    # Restart from best with perturbation
                    active = best_active[:]
                    k_count = [0] * num_k222s
                    for t_id in range(num_triples):
                        if active[t_id]:
                            for k_idx in triple_to_k222s[t_id]:
                                k_count[k_idx] += 1
                    current_size = best_size
                    # Perturb: remove a few random active triples
                    for _ in range(rng.randint(1, 3)):
                        t_remove = rng.randint(0, num_triples - 1)
                        if active[t_remove]:
                            active[t_remove] = False
                            current_size -= 1
                            for k_idx in triple_to_k222s[t_remove]:
                                k_count[k_idx] -= 1
        
        # Final polish: try adding any triple that fits
        for t_id in range(num_triples):
            if not best_active[t_id]:
                ok = True
                # Reconstruct k_count for best_active
                pass # Skip for speed; the local search already polishes
        
        return best_size, best_active

    best_size = 0
    best_active = None
    
    seed = 0
    while time.time() < DEADLINE:
        size, active = solve_instance(seed)
        if size > best_size:
            best_size = size
            best_active = active[:]
        seed += 1
        
        # Polish the best solution periodically
        if seed % 3 == 0 and best_active and time.time() < DEADLINE - 5:
            # Reconstruct k_count for best_active
            k_count = [0] * num_k222s
            for t_id in range(num_triples):
                if best_active[t_id]:
                    for k_idx in triple_to_k222s[t_id]:
                        k_count[k_idx] += 1
            
            # Try to add any triple that fits
            improved = True
            while improved:
                improved = False
                for t_id in range(num_triples):
                    if not best_active[t_id]:
                        ok = True
                        for k_idx in triple_to_k222s[t_id]:
                            if k_count[k_idx] >= 7:
                                ok = False
                                break
                        if ok:
                            best_active[t_id] = True
                            best_size += 1
                            for k_idx in triple_to_k222s[t_id]:
                                k_count[k_idx] += 1
                            improved = True
    
    if best_active is None:
        return []
        
    return [list(triples[i]) for i in range(num_triples) if best_active[i]]
