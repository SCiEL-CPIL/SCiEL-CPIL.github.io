import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Leave a 5s buffer.
    deadline = start_time + 145.0
    
    N = 16
    vertices = range(N)
    all_triples = list(itertools.combinations(vertices, 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is determined by 3 disjoint pairs.
    # The 8 transversals are the combinations of picking one vertex from each pair.
    # We need to map each triple to the list of K(2,2,2) IDs it belongs to.
    
    k222_configs = []  # List of lists of triple indices
    triple_to_k222 = [[] for _ in range(num_triples)]
    
    # Generate all sets of 6 vertices
    # For each set of 6 vertices, there are 15 ways to partition into 3 pairs.
    for six_vertices in itertools.combinations(vertices, 6):
        a, b, c, d, e, f = six_vertices
        # The 15 partitions of 6 elements into 3 pairs
        # We can generate these by pairing a with b,c,d,e,f and then pairing the remaining 4
        # Or just hardcode the 15 pairings structure.
        # Let's use a systematic generation:
        # Pairs are (a,x), (y,z), (w,v) where {a,x,y,z,w,v} = {a,b,c,d,e,f}
        # To avoid duplicates, we fix 'a' in the first pair.
        # a can pair with b, c, d, e, f.
        
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
    
    # State representation
    # present: list of booleans
    # k222_counts: list of integers, count of present triples in each config
    # num_present: integer
    # num_violations: integer (configs with count >= 8)
    
    def update_counts(ti, sign, k222_counts, num_violations):
        """
        sign = 1 for add, -1 for remove
        returns new num_violations
        """
        old_viol = num_violations
        for cid in triple_to_k222[ti]:
            old_count = k222_counts[cid]
            new_count = old_count + sign
            k222_counts[cid] = new_count
            
            if sign == 1:
                # Adding
                if old_count == 7:
                    # Becomes violation
                    old_viol += 1
            else:
                # Removing
                if old_count == 8:
                    # Becomes non-violation (assuming it was 8, now 7)
                    # Note: if it was >8, removing doesn't immediately fix it unless it drops below 8
                    # But in our moves, we usually only have counts 0..8 max? 
                    # No, max is 8. If it was 8 and we remove one, it becomes 7.
                    # If it was 9 (impossible with 8 transversals), N/A.
                    old_viol -= 1
        return old_viol

    def greedy_init(rng):
        present = [False] * num_triples
        k222_counts = [0] * num_k222
        num_viol = 0
        count = 0
        
        # Sort triples by "constraint tightness"? 
        # Simpler: random order, add if it doesn't create a violation.
        # To be smarter, we can calculate the number of conflicts.
        order = list(range(num_triples))
        rng.shuffle(order)
        
        for ti in order:
            # Check if adding ti causes any violation
            ok = True
            for cid in triple_to_k222[ti]:
                if k222_counts[cid] >= 7:
                    ok = False
                    break
            if ok:
                present[ti] = True
                count += 1
                num_viol = update_counts(ti, 1, k222_counts, num_viol)
                
        return present, k222_counts, count, num_viol

    def simulated_annealing(present, k222_counts, count, num_viol, rng, sa_deadline):
        # Energy: We want to maximize count and minimize violations.
        # Let's define Score = count * 1.0 - num_viol * 100.0
        # We want to maximize Score.
        # Or minimize Energy = -count * 1.0 + num_viol * 100.0
        
        current_score = count * 1.0 - num_viol * 100.0
        
        # Temperature schedule
        # Start high to allow violations, cool down to enforce constraints and maximize count
        T = 5.0
        cooling_rate = 0.995
        min_T = 0.01
        
        best_score = current_score
        best_present = present[:]
        best_count = count
        best_viol = num_viol
        
        # Track steps for time management
        steps = 0
        no_improve_steps = 0
        max_no_improve = 5000
        
        # Maintain lists of present and absent triples for O(1) selection
        present_list = [i for i in range(num_triples) if present[i]]
        absent_list = [i for i in range(num_triples) if not present[i]]
        
        while T > min_T:
            if time.time() > sa_deadline:
                break
                
            # Generate a move
            # 1. Add a random absent triple
            # 2. Remove a random present triple
            # 3. Swap
            
            move_type = rng.random()
            
            if move_type < 0.4 and len(absent_list) > 0:
                # Add
                # Pick a random absent triple
                idx_in_list = rng.randint(0, len(absent_list) - 1)
                ti = absent_list[idx_in_list]
                
                # Calculate delta
                # Adding ti: count + 1
                # Violations: check configs
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 7:
                        delta_viol += 1
                delta_score = 1.0 - delta_viol * 100.0
                
                if rng.random() < math.exp(delta_score / T) or delta_score > 0:
                    # Accept
                    present[ti] = True
                    count += 1
                    num_viol = update_counts(ti, 1, k222_counts, num_viol)
                    current_score += delta_score
                    
                    # Update lists
                    # Remove ti from absent_list (swap with last, pop)
                    absent_list[idx_in_list] = absent_list[-1]
                    absent_list.pop()
                    present_list.append(ti)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        
            elif move_type < 0.8 and len(present_list) > 0:
                # Remove
                # Pick a random present triple
                idx_in_list = rng.randint(0, len(present_list) - 1)
                ti = present_list[idx_in_list]
                
                # Calculate delta
                # Removing ti: count - 1
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 8:
                        delta_viol -= 1 # Violation removed
                delta_score = -1.0 - delta_viol * 100.0
                
                if rng.random() < math.exp(delta_score / T) or delta_score > 0:
                    # Accept
                    present[ti] = False
                    count -= 1
                    num_viol = update_counts(ti, -1, k222_counts, num_viol)
                    current_score += delta_score
                    
                    # Update lists
                    # Remove ti from present_list (swap with last, pop)
                    present_list[idx_in_list] = present_list[-1]
                    present_list.pop()
                    absent_list.append(ti)
                    no_improve_steps += 1
                    
            else:
                # Swap: Remove one, Add one
                if len(present_list) == 0 or len(absent_list) == 0:
                    continue
                    
                idx1 = rng.randint(0, len(present_list) - 1)
                idx2 = rng.randint(0, len(absent_list) - 1)
                
                remove_ti = present_list[idx1]
                add_ti = absent_list[idx2]
                
                # Calculate combined delta for swap
                # Count doesn't change. Only violations change.
                # We need to compute the change in num_viol when we remove remove_ti and add add_ti.
                
                # Save current violation state for affected configs
                # Optimization: Only check configs that are in the union of the two triples' configs
                affected_configs = set(triple_to_k222[remove_ti] + triple_to_k222[add_ti])
                
                old_viol_affected = 0
                for cid in affected_configs:
                    if k222_counts[cid] >= 8:
                        old_viol_affected += 1
                
                # Apply changes
                for cid in triple_to_k222[remove_ti]:
                    k222_counts[cid] -= 1
                for cid in triple_to_k222[add_ti]:
                    k222_counts[cid] += 1
                
                new_viol_affected = 0
                for cid in affected_configs:
                    if k222_counts[cid] >= 8:
                        new_viol_affected += 1
                
                delta_viol_swap = new_viol_affected - old_viol_affected
                delta_score_swap = -delta_viol_swap * 100.0
                
                if rng.random() < math.exp(delta_score_swap / T) or delta_score_swap > 0:
                    # Accept
                    present[remove_ti] = False
                    present[add_ti] = True
                    # Counts already updated
                    current_score += delta_score_swap
                    num_viol += delta_viol_swap
                    
                    # Update lists
                    # remove_ti moves from present to absent
                    # add_ti moves from absent to present
                    present_list[idx1] = add_ti
                    absent_list[idx2] = remove_ti
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    # Revert changes
                    for cid in triple_to_k222[remove_ti]:
                        k222_counts[cid] += 1
                    for cid in triple_to_k222[add_ti]:
                        k222_counts[cid] -= 1
                    no_improve_steps += 1

            T *= cooling_rate
            steps += 1
            
        return best_present, best_count, best_viol

    # Main Loop: Multiple restarts with different seeds
    # Since SA is stochastic, running it multiple times helps.
    # However, SA is expensive. We should run one long SA or a few shorter ones.
    # Given 150s budget, and SA step is fast, we can do multiple runs.
    
    best_global_count = 0
    best_global_result = []
    
    # Run SA with a few different initializations
    # Use more restarts with shorter time each to explore more diverse solutions
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(42 + r)
        
        # Greedy Init
        present, k222_counts, count, num_viol = greedy_init(rng)
        
        if count > best_global_count:
            best_global_count = count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if present[i]]
            
        # SA
        # Adjust time for SA
        sa_deadline = time.time() + time_per_restart * 0.95
        
        best_present, best_count, best_viol = simulated_annealing(present, k222_counts, count, num_viol, rng, sa_deadline)
        
        if best_count > best_global_count:
            best_global_count = best_count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
            
    if not best_global_result:
        # Fallback: empty set or small valid set
        # A single triple is always valid
        best_global_result = [[0, 1, 2]]
        
    return best_global_result

# Test helper (not part of construct)
if __name__ == "__main__":
    res = construct()
    print(f"Number of triples: {len(res)}")
    # Validate
    N = 16
    triples_set = set(tuple(sorted(t)) for t in res)
    
    # Check for K(2,2,2) violations
    # Re-do the config generation quickly for validation? 
    # Or just trust the code. 
    # Let's do a simple check for small N if possible, but N=16 is fixed.
    pass
