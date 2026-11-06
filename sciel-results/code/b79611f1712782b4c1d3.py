import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all possible triples
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a, b, c))
    
    # Precompute intersection sizes between all pairs of triples
    # For efficiency, represent each triple as a frozenset
    triple_sets = [frozenset(t) for t in all_triples]
    
    def are_compatible(t1, t2):
        """Check if two triples share at most one point."""
        # t1, t2 are frozensets
        inter = len(t1 & t2)
        return inter <= 1
    
    def is_valid_system(triples_list):
        """Check if a list of triples forms a linear system."""
        sets = [frozenset(t) for t in triples_list]
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                if len(sets[i] & sets[j]) > 1:
                    return False
        return True
    
    def get_independence_number(triples_list):
        """Find the largest subset of points containing no triple."""
        if not triples_list:
            return n
        
        # Build a set of frozensets for fast lookup
        triple_set = frozenset(triples_list)
        
        best = 0
        # Check all subsets by bitmask
        # For each subset, check if it contains any triple
        # A subset S contains triple T if T is a subset of S
        # We can iterate over all triples and mark subsets that contain them
        # But with 65536 subsets and up to 37 triples, direct check is fine
        
        # Alternative: for each mask from 0 to 2^n-1, check if any triple is contained
        # This is O(2^n * |triples|) which is 65536 * 37 ≈ 2.4M, fine
        
        for mask in range(1, 1 << n):
            # Check if this subset contains any triple
            has_triple = False
            for t in triples_list:
                # Check if all three points are in the mask
                if (mask & (1 << t[0])) and (mask & (1 << t[1])) and (mask & (1 << t[2])):
                    has_triple = True
                    break
            if not has_triple:
                cnt = bin(mask).count('1')
                if cnt > best:
                    best = cnt
                    if best == n:  # Can't do better
                        break
        
        return best
    
    # Greedy construction to get a good initial solution
    def greedy_build():
        """Build a maximal set of compatible triples greedily."""
        chosen = []
        chosen_sets = []
        random.shuffle(all_triples)
        for t in all_triples:
            tset = frozenset(t)
            compatible = True
            for cs in chosen_sets:
                if len(tset & cs) > 1:
                    compatible = False
                    break
            if compatible:
                chosen.append(t)
                chosen_sets.append(tset)
        return chosen, chosen_sets
    
    # Simulated annealing
    def simulated_annealing(initial_triples, initial_sets, max_iters=500000):
        best_triples = list(initial_triples)
        best_sets = list(initial_sets)
        best_count = len(best_triples)
        best_alpha = get_independence_number(best_triples)
        
        current_triples = list(initial_triples)
        current_sets = list(initial_sets)
        current_count = len(current_triples)
        
        T = 10.0
        cooling = 0.9995
        T_min = 0.01
        iters = 0
        
        while iters < max_iters:
            if time.time() > deadline - 10:
                break
            
            iters += 1
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4 and current_count > 0:
                # Remove a random triple
                idx = random.randint(0, current_count - 1)
                removed = current_triples[idx]
                removed_set = current_sets[idx]
                current_triples.pop(idx)
                current_sets.pop(idx)
                current_count -= 1
                
                # Evaluate
                if current_count > best_count:
                    best_triples = list(current_triples)
                    best_sets = list(current_sets)
                    best_count = current_count
                    best_alpha = get_independence_number(best_triples)
                
                # Accept or reject removal
                # We generally want to keep more triples, so reject removal with high prob
                if random.random() < 0.3:
                    # Reject: put it back
                    current_triples.append(removed)
                    current_sets.append(removed_set)
                    current_count += 1
                    
            elif move_type < 0.8 and current_count > 0:
                # Swap: remove one, add another
                idx = random.randint(0, current_count - 1)
                removed = current_triples[idx]
                removed_set = current_sets[idx]
                current_triples.pop(idx)
                current_sets.pop(idx)
                current_count -= 1
                
                # Try to add a random new triple
                added = None
                added_set = None
                for _ in range(50):  # Try up to 50 random candidates
                    candidate = random.choice(all_triples)
                    cset = frozenset(candidate)
                    compatible = True
                    for cs in current_sets:
                        if len(cset & cs) > 1:
                            compatible = False
                            break
                    if compatible:
                        added = candidate
                        added_set = cset
                        break
                
                if added is not None:
                    current_triples.append(added)
                    current_sets.append(added_set)
                    current_count += 1
                    
                    if current_count > best_count:
                        best_triples = list(current_triples)
                        best_sets = list(current_sets)
                        best_count = current_count
                        best_alpha = get_independence_number(best_triples)
                else:
                    # Couldn't add, put removed back
                    current_triples.append(removed)
                    current_sets.append(removed_set)
                    current_count += 1
                    
            else:
                # Try to add a new triple
                added = None
                added_set = None
                for _ in range(100):
                    candidate = random.choice(all_triples)
                    cset = frozenset(candidate)
                    compatible = True
                    for cs in current_sets:
                        if len(cset & cs) > 1:
                            compatible = False
                            break
                    if compatible:
                        added = candidate
                        added_set = cset
                        break
                
                if added is not None:
                    current_triples.append(added)
                    current_sets.append(added_set)
                    current_count += 1
                    
                    if current_count > best_count:
                        best_triples = list(current_triples)
                        best_sets = list(current_sets)
                        best_count = current_count
                        best_alpha = get_independence_number(best_triples)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_triples, best_alpha
    
    # Run multiple restarts
    best_triples_overall = None
    best_alpha_overall = n + 1
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        random.seed(r * 12345 + 678)
        init_triples, init_sets = greedy_build()
        
        # Run SA
        sa_triples, sa_alpha = simulated_annealing(init_triples, init_sets, max_iters=200000)
        
        if sa_alpha < best_alpha_overall:
            best_alpha_overall = sa_alpha
            best_triples_overall = list(sa_triples)
        
        # Also try the initial greedy solution
        if len(init_triples) > (len(best_triples_overall) if best_triples_overall else 0):
            greedy_alpha = get_independence_number(init_triples)
            if greedy_alpha < best_alpha_overall:
                best_alpha_overall = greedy_alpha
                best_triples_overall = list(init_triples)
    
    if best_triples_overall is None or len(best_triples_overall) == 0:
        # Fallback to seed
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    # Convert to list of lists
    result = [list(t) for t in best_triples_overall]
    
    # Final verification
    assert is_valid_system(result), "Invalid linear system produced"
    
    return result
