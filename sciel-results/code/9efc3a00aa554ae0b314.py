import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute conflicts: for each x, list of pairs (a,b) such that
    # gcd(a,b) == gcd(a,x) == gcd(b,x), with a < b, a != x, b != x
    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            ga_x = gcds[a][x]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    conflicts[x].add((a, b))
    
    # Convert to lists for faster iteration
    for x in range(1, N + 1):
        conflicts[x] = list(conflicts[x])
    
    # Precompute for each x: the set of elements that conflict with x in some pair
    # conflict_partners[x] = set of all a,b that appear in any conflict pair for x
    conflict_partners = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a, b in conflicts[x]:
            conflict_partners[x].add(a)
            conflict_partners[x].add(b)
    
    def is_valid_addition(S_set, x):
        """Check if x can be added to S_set without creating a bad triple."""
        # Quick check: if any conflict partner of x is in S_set, we need to check pairs
        # But we still need to check all pairs
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
    best_set = set()
    best_size = 0
    
    def greedy_init(seed_offset=0):
        """Greedy initialization with random ordering."""
        S = set()
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            if is_valid_addition(S, x):
                S.add(x)
        return S
    
    def tabu_search(S, max_iter=5000):
        """Tabu search to improve the set."""
        S = set(S)
        current_size = len(S)
        tabu_list = {}  # element -> iteration when it can be added again
        tabu_tenure = 20
        no_improve = 0
        local_best_size = current_size
        local_best_set = set(S)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add an element
            not_in_S = [x for x in elements if x not in S]
            if not_in_S:
                random.shuffle(not_in_S)
                added = False
                for x in not_in_S:
                    if is_valid_addition(S, x):
                        # Check tabu status with aspiration
                        if x in tabu_list and tabu_list[x] > i:
                            # Tabu, but allow if it improves best
                            if current_size + 1 > local_best_size:
                                S.add(x)
                                current_size += 1
                                if current_size > local_best_size:
                                    local_best_size = current_size
                                    local_best_set = set(S)
                                no_improve = 0
                                added = True
                                break
                        else:
                            S.add(x)
                            current_size += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_set = set(S)
                            no_improve = 0
                            added = True
                            break
                
                if added:
                    continue
            
            # If no add possible, try a 1-for-2 swap: remove u, add v and w
            swap_found = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                
                # S_temp = S - {u}
                S_temp = S - {u}
                
                # Find v that is valid with S_temp
                candidates = [x for x in elements if x not in S]
                random.shuffle(candidates)
                
                v_found = None
                for v in candidates:
                    if is_valid_addition(S_temp, v):
                        v_found = v
                        break
                
                if v_found is not None:
                    # Find w that is valid with S_temp + {v}
                    S_temp2 = S_temp | {v_found}
                    w_found = None
                    for w in candidates:
                        if w == v_found:
                            continue
                        if is_valid_addition(S_temp2, w):
                            w_found = w
                            break
                    
                    if w_found is not None:
                        # Perform swap: remove u, add v and w
                        S = S_temp2 | {w_found}
                        current_size += 1
                        tabu_list[u] = i + tabu_tenure
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_set = set(S)
                        no_improve = 0
                        swap_found = True
                        break
            
            if swap_found:
                continue
            
            # If no swap found, try perturbation: remove a random element
            if no_improve > 20:
                if S:
                    u = random.choice(list(S))
                    S.remove(u)
                    current_size -= 1
                    tabu_list[u] = i + tabu_tenure
                    no_improve = 0
            else:
                no_improve += 1
                
        return local_best_set
    
    # Run multiple restarts
    for run in range(200):
        if time.time() > deadline - 5:
            break
            
        S = greedy_init()
        S = tabu_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final polish: try to add any remaining elements
    not_in_best = [x for x in elements if x not in best_set]
    random.shuffle(not_in_best)
    for x in not_in_best:
        if is_valid_addition(best_set, x):
            best_set.add(x)
    
    return list(best_set)
