import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 16
    k = 4
    all_sets = list(combinations(range(1, n+1), k))
    all_set_objs = [frozenset(s) for s in all_sets]
    N = len(all_set_objs)
    
    def check_conflict(new_set, family):
        """Returns (conflict_bool, (A,B) if conflict)."""
        groups = {}
        for A in family:
            inter = A & new_set
            groups.setdefault(inter, []).append(A)
        
        for S, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    if group[i] & group[j] == S:
                        return True, (group[i], group[j])
        return False, None
    
    random.seed(12345)
    
    best_family = []
    best_size = 0
    
    # Phase 1: Multiple greedy starts
    max_greedy_trials = 30
    for trial in range(max_greedy_trials):
        if time.time() - start_time > DEADLINE:
            break
        
        family = []
        shuffled = list(range(N))
        random.shuffle(shuffled)
        
        for idx in shuffled:
            if time.time() - start_time > DEADLINE:
                break
            s = all_set_objs[idx]
            conflict, _ = check_conflict(s, family)
            if not conflict:
                family.append(s)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    # Phase 2: Local search with swaps from best solution
    family = list(best_family)
    family_set = set(family)
    
    local_search_iterations = 0
    max_local_iters = 20000
    
    while local_search_iterations < max_local_iters:
        if time.time() - start_time > DEADLINE:
            break
        local_search_iterations += 1
        
        # Try to add a random set
        idx = random.randint(0, N-1)
        s = all_set_objs[idx]
        if s in family_set:
            continue
        
        conflict, conflicting_pair = check_conflict(s, family)
        if not conflict:
            family.append(s)
            family_set.add(s)
            if len(family) > best_size:
                best_size = len(family)
                best_family = list(family)
        else:
            # Try swap: remove one of the conflicting sets
            A, B = conflicting_pair
            removed = False
            for remove_set in [A, B]:
                if time.time() - start_time > DEADLINE:
                    break
                new_family = [x for x in family if x != remove_set]
                conflict2, _ = check_conflict(s, new_family)
                if not conflict2:
                    family = new_family
                    family_set = set(family)
                    family.append(s)
                    family_set.add(s)
                    if len(family) > best_size:
                        best_size = len(family)
                        best_family = list(family)
                    removed = True
                    break
            
            # If no swap worked, occasionally perturb
            if not removed and random.random() < 0.005 and len(family) > 15:
                # Remove 2-3 random sets to create space
                num_remove = random.randint(2, 3)
                for _ in range(num_remove):
                    if family:
                        rem_idx = random.randint(0, len(family)-1)
                        family.pop(rem_idx)
                family_set = set(family)
    
    return [sorted(s) for s in best_family]
