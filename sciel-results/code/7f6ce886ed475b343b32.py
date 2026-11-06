import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 16
    vertices = range(N)
    all_triples = list(itertools.combinations(vertices, 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is determined by 3 disjoint pairs.
    # The 8 transversals are the combinations of picking one vertex from each pair.
    
    k222_configs = []  # List of lists of triple indices
    triple_to_k222 = [[] for _ in range(num_triples)]
    
    # Generate all sets of 6 vertices
    for six_vertices in itertools.combinations(vertices, 6):
        a, b, c, d, e, f = six_vertices
        pairs_list = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        
        for p1, p2, p3 in pairs_list:
            config_triples = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        idx = triple_to_idx[t]
                        config_triples.append(idx)
            
            config_id = len(k222_configs)
            k222_configs.append(config_triples)
            
            for t_idx in config_triples:
                triple_to_k222[t_idx].append(config_id)
                
    num_k222 = len(k222_configs)
    
    # Precompute bitmasks for configurations
    # config_mask[cid] is an integer with bits set for the 8 triples in config cid
    config_masks = [0] * num_k222
    for cid, config in enumerate(k222_configs):
        mask = 0
        for t_idx in config:
            mask |= (1 << t_idx)
        config_masks[cid] = mask
            
    # Precompute "other 7" masks for fast addition check
    # For each triple t and each config c containing t, 
    # other_mask[cid, t] is the mask of c without t
    # We can store this in a list of lists: triple_to_other_masks[t] = [mask_c_without_t for c in triple_to_k222[t]]
    triple_to_other_masks = [[] for _ in range(num_triples)]
    for t in range(num_triples):
        for cid in triple_to_k222[t]:
            full_mask = config_masks[cid]
            t_bit = 1 << t
            other_mask = full_mask & ~t_bit
            triple_to_other_masks[t].append(other_mask)

    def greedy_init(rng):
        is_present = [False] * num_triples
        present_mask = 0
        config_counts = [0] * num_k222
        count = 0
        
        order = list(range(num_triples))
        rng.shuffle(order)
        
        for ti in order:
            # Check if adding ti is valid
            # Valid if for all c in triple_to_k222[ti], 
            # (present_mask & other_mask) != other_mask
            ok = True
            for om in triple_to_other_masks[ti]:
                if (present_mask & om) == om:
                    ok = False
                    break
            if ok:
                is_present[ti] = True
                present_mask |= (1 << ti)
                count += 1
                for cid in triple_to_k222[ti]:
                    config_counts[cid] += 1
                    
        return is_present, present_mask, config_counts, count

    def simulated_annealing(is_present, present_mask, config_counts, count, rng, sa_deadline):
        current_score = count * 1.0 - 0.0 # We track violations separately or include in score?
        # Let's use Score = count - 100 * violations
        violations = 0
        for c in config_counts:
            if c >= 8:
                violations += 1
                
        current_score = count - 100.0 * violations
        
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.01
        
        best_score = current_score
        best_is_present = is_present[:]
        best_count = count
        best_violations = violations
        best_mask = present_mask
        
        no_improve_steps = 0
        max_no_improve = 20000 # Increase this to allow longer runs
        
        while T > min_T:
            if time.time() > sa_deadline:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Add a random absent triple
                ti = rng.randint(0, num_triples - 1)
                if is_present[ti]:
                    continue
                
                # Calculate delta
                # count + 1
                # violations change:
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if config_counts[cid] == 7:
                        delta_viol += 1
                    elif config_counts[cid] == 8:
                        # Already a violation, adding another doesn't increase violation count?
                        # Wait, violation is defined as "config is full".
                        # If it's already full, it's still full.
                        # But can a config have > 8 triples present? No, it only has 8 triples total.
                        # So if count is 8, it's a violation. Adding a triple NOT in this config doesn't affect it.
                        # Adding a triple IN this config? It's already present? No, ti is absent.
                        # If ti is in config cid, and config_counts[cid] is 8, that's impossible because ti is absent.
                        # So if ti is in cid, config_counts[cid] <= 7.
                        pass
                
                delta_score = 1.0 - 100.0 * delta_viol
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    # Accept
                    is_present[ti] = True
                    present_mask |= (1 << ti)
                    count += 1
                    for cid in triple_to_k222[ti]:
                        config_counts[cid] += 1
                        if config_counts[cid] == 8:
                            violations += 1
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_is_present = is_present[:]
                        best_count = count
                        best_violations = violations
                        best_mask = present_mask
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        
            elif move_type < 0.9:
                # Remove a random present triple
                ti = rng.randint(0, num_triples - 1)
                if not is_present[ti]:
                    continue
                
                # Calculate delta
                # count - 1
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if config_counts[cid] == 8:
                        delta_viol -= 1 # Violation removed
                
                delta_score = -1.0 - 100.0 * delta_viol
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    # Accept
                    is_present[ti] = False
                    present_mask &= ~(1 << ti)
                    count -= 1
                    for cid in triple_to_k222[ti]:
                        config_counts[cid] -= 1
                        if config_counts[cid] == 7: # Was 8, now 7
                            violations -= 1
                    current_score += delta_score
                    no_improve_steps += 1
                    
            else:
                # Swap: Remove one, Add one
                ti1 = rng.randint(0, num_triples - 1)
                ti2 = rng.randint(0, num_triples - 1)
                if ti1 == ti2:
                    continue
                if is_present[ti1] == is_present[ti2]:
                    continue
                
                if is_present[ti1]:
                    remove_ti = ti1
                    add_ti = ti2
                else:
                    remove_ti = ti2
                    add_ti = ti1
                    
                # Count doesn't change.
                # Check violations.
                # We need to check if the state after swap is valid or has fewer violations.
                # Affected configs: those containing remove_ti or add_ti.
                
                # Calculate delta violations carefully.
                # We simulate the change in counts for affected configs.
                
                affected_configs = set(triple_to_k222[remove_ti] + triple_to_k222[add_ti])
                
                old_viol_affected = 0
                for cid in affected_configs:
                    if config_counts[cid] >= 8:
                        old_viol_affected += 1
                
                # Apply changes temporarily to counts? 
                # Or just calculate new counts.
                # new_count[cid] = config_counts[cid] - (1 if remove_ti in cid else 0) + (1 if add_ti in cid else 0)
                
                new_viol_affected = 0
                for cid in affected_configs:
                    new_c = config_counts[cid]
                    if remove_ti in triple_to_k222[remove_ti]: # This is always true, but we need to check if remove_ti is in cid
                        # Actually, triple_to_k222[remove_ti] gives all configs containing remove_ti.
                        # So if cid is in that list, remove_ti is in cid.
                        pass
                    
                    # Let's use a helper or just check membership.
                    # Since lists are small, we can precompute membership? 
                    # Or just:
                    # is_remove_in = (cid in triple_to_k222[remove_ti])
                    # is_add_in = (cid in triple_to_k222[add_ti])
                    
                    # To speed this up, we can use the fact that we are iterating over affected_configs.
                    # But checking `cid in list` is O(K).
                    # Alternative: 
                    # We know the configs for remove_ti and add_ti.
                    # We can just iterate through them and update a dict of counts?
                    # No, just calculate:
                    
                    new_c = config_counts[cid]
                    if remove_ti in triple_to_k222[remove_ti]: # Logic error, should be: if remove_ti is in config cid
                        # We need a way to check if ti is in config cid quickly.
                        # We can use the bitmask!
                        # if (config_masks[cid] >> remove_ti) & 1:
                        # if (config_masks[cid] >> add_ti) & 1:
                        pass
                    
                    if (config_masks[cid] >> remove_ti) & 1:
                        new_c -= 1
                    if (config_masks[cid] >> add_ti) & 1:
                        new_c += 1
                        
                    if new_c >= 8:
                        new_viol_affected += 1
                        
                delta_viol_swap = new_viol_affected - old_viol_affected
                delta_score_swap = -100.0 * delta_viol_swap
                
                if delta_score_swap > 0 or rng.random() < math.exp(delta_score_swap / T):
                    # Accept
                    # Update state
                    is_present[remove_ti] = False
                    is_present[add_ti] = True
                    present_mask &= ~(1 << remove_ti)
                    present_mask |= (1 << add_ti)
                    
                    # Update counts
                    for cid in triple_to_k222[remove_ti]:
                        config_counts[cid] -= 1
                    for cid in triple_to_k222[add_ti]:
                        config_counts[cid] += 1
                    
                    violations += delta_viol_swap
                    current_score += delta_score_swap
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_is_present = is_present[:]
                        best_count = count
                        best_violations = violations
                        best_mask = present_mask
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
                    
            T *= cooling_rate
            if no_improve_steps > max_no_improve:
                # Restart temperature? Or break?
                # Let's just continue, cooling will handle it.
                pass
                
        # Ensure the best solution is valid (violations == 0)
        # If best_violations > 0, we might need to clean it up.
        # But SA usually finds valid solutions if penalty is high.
        # Let's verify and clean if necessary.
        
        if best_violations > 0:
            # Greedy cleanup: remove triples causing violations
            # This is a fallback
            is_p = best_is_present[:]
            pm = best_mask
            cc = [0] * num_k222
            cnt = 0
            for i in range(num_triples):
                if is_p[i]:
                    cnt += 1
                    for cid in triple_to_k222[i]:
                        cc[cid] += 1
            
            # Remove triples that are part of violations
            changed = True
            while changed:
                changed = False
                for i in range(num_triples):
                    if is_p[i]:
                        for cid in triple_to_k222[i]:
                            if cc[cid] >= 8:
                                # Remove i
                                is_p[i] = False
                                pm &= ~(1 << i)
                                cnt -= 1
                                for c2 in triple_to_k222[i]:
                                    cc[c2] -= 1
                                changed = True
                                break
                    if not is_p[i]:
                        continue
                            
            best_is_present = is_p
            best_count = cnt
            best_violations = 0 # Should be 0 now
            
        return best_is_present, best_count

    best_global_count = 0
    best_global_result = []
    
    # Run SA with multiple restarts
    num_restarts = 30
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(42 + r)
        
        # Greedy Init
        is_present, present_mask, config_counts, count = greedy_init(rng)
        
        if count > best_global_count:
            best_global_count = count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if is_present[i]]
            
        # SA
        sa_deadline = time.time() + time_per_restart * 0.95
        best_is_present, best_count = simulated_annealing(is_present, present_mask, config_counts, count, rng, sa_deadline)
        
        if best_count > best_global_count:
            best_global_count = best_count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if best_is_present[i]]
            
    if not best_global_result:
        best_global_result = [[0, 1, 2]]
        
    return best_global_result

if __name__ == "__main__":
    res = construct()
    print(f"Number of triples: {len(res)}")
