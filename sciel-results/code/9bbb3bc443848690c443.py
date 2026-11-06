import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            ap3s.append((a, a+d, a+2*d))
            
    ap4s = []
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            ap4s.append((a, a+d, a+2*d, a+3*d))
            
    # Map element to AP indices
    elem_to_ap3 = [[] for _ in range(N+1)]
    for i, (a, b, c) in enumerate(ap3s):
        elem_to_ap3[a].append(i)
        elem_to_ap3[b].append(i)
        elem_to_ap3[c].append(i)
        
    elem_to_ap4 = [[] for _ in range(N+1)]
    for i, (a, b, c, d) in enumerate(ap4s):
        elem_to_ap4[a].append(i)
        elem_to_ap4[b].append(i)
        elem_to_ap4[c].append(i)
        elem_to_ap4[d].append(i)
    
    # Precompute for each element, the list of (other1, other2) pairs for 3-APs
    # and the list of (other1, other2, other3) tuples for 4-APs
    # This allows fast delta computation
    elem_3ap_others = [[] for _ in range(N+1)]
    for i, (a, b, c) in enumerate(ap3s):
        elem_3ap_others[a].append((b, c))
        elem_3ap_others[b].append((a, c))
        elem_3ap_others[c].append((a, b))
        
    elem_4ap_others = [[] for _ in range(N+1)]
    for i, (a, b, c, d) in enumerate(ap4s):
        elem_4ap_others[a].append((b, c, d))
        elem_4ap_others[b].append((a, c, d))
        elem_4ap_others[c].append((a, b, d))
        elem_4ap_others[d].append((a, b, c))
    
    def count_3aps(S):
        """Count 3-APs in set S"""
        count = 0
        for a, b, c in ap3s:
            if a in S and b in S and c in S:
                count += 1
        return count
    
    def is_add_valid(S, y):
        """Check if adding y to S creates a 4-AP"""
        for others in elem_4ap_others[y]:
            if all(o in S for o in others):
                return False
        return True
    
    def is_swap_valid(S, x, y):
        """Check if swapping x for y in S is valid (no 4-AP)"""
        # After swap, S' = (S - {x}) ∪ {y}
        # A 4-AP is formed if there's a 4-AP containing y where other 3 are in S - {x}
        for others in elem_4ap_others[y]:
            # All others must be in S and none can be x
            if x in others:
                continue
            if all(o in S for o in others):
                return False
        return True
    
    def delta_add(S, y):
        """Compute delta in 3-AP count when adding y to S"""
        delta = 0
        for o1, o2 in elem_3ap_others[y]:
            if o1 in S and o2 in S:
                delta += 1
        return delta
    
    def delta_remove(S, x):
        """Compute delta in 3-AP count when removing x from S"""
        delta = 0
        for o1, o2 in elem_3ap_others[x]:
            if o1 in S and o2 in S:
                delta += 1
        return delta
    
    def greedy_init(seed=None):
        """Generate initial solution using greedy with random ordering"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        S = set()
        candidates = list(range(1, N+1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if is_add_valid(S, x):
                S.add(x)
        
        return S
    
    def local_search(S, max_time=5.0):
        """Perform local search with add, remove, and swap moves"""
        S = set(S)
        score = count_3aps(S)
        start_ls = time.time()
        
        improved = True
        while improved and time.time() - start_ls < max_time:
            improved = False
            
            # Try additions
            best_add = None
            best_add_delta = 0
            for x in range(1, N+1):
                if x not in S:
                    if is_add_valid(S, x):
                        d = delta_add(S, x)
                        if d > best_add_delta:
                            best_add_delta = d
                            best_add = x
            
            if best_add is not None:
                S.add(best_add)
                score += best_add_delta
                improved = True
                continue
            
            # Try swaps
            best_swap = None
            best_swap_delta = 0
            S_list = list(S)
            for x in S_list:
                # Sample candidates for y
                for _ in range(15):
                    y = random.randint(1, N)
                    if y in S:
                        continue
                    if not is_swap_valid(S, x, y):
                        continue
                    
                    # Calculate delta
                    rem_delta = -delta_remove(S, x)
                    # For add delta, we need to check against S - {x}
                    add_delta = 0
                    for o1, o2 in elem_3ap_others[y]:
                        if o1 == x or o2 == x:
                            continue
                        if o1 in S and o2 in S:
                            add_delta += 1
                    
                    total_delta = rem_delta + add_delta
                    if total_delta > best_swap_delta:
                        best_swap_delta = total_delta
                        best_swap = (x, y)
            
            if best_swap is not None:
                x, y = best_swap
                S.remove(x)
                S.add(y)
                score += best_swap_delta
                improved = True
                continue
            
            # If no add, no swap, try a random removal to escape local optimum
            if len(S) > 0:
                x = random.choice(list(S))
                S.remove(x)
                score += -delta_remove(S, x)
                improved = True
                continue
                
        return S, score
    
    def simulated_annealing(S, max_time=10.0):
        """Simulated annealing with geometric cooling"""
        S = set(S)
        current_score = count_3aps(S)
        best_S = set(S)
        best_score = current_score
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        T = T_start
        
        not_in_S = set(range(1, N+1)) - S
        
        start_sa = time.time()
        cooling_rate = 0.9995
        
        while time.time() - start_sa < max_time:
            # Choose move type: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and len(S) < 200:
                # Add a random element
                if not not_in_S:
                    continue
                y = random.choice(list(not_in_S))
                if not is_add_valid(S, y):
                    continue
                
                delta = delta_add(S, y)
                
                # Metropolis acceptance
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S.add(y)
                    current_score += delta
                    not_in_S.remove(y)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        
            elif move_type < 0.5 and len(S) > 50:
                # Remove a random element
                x = random.choice(list(S))
                delta = -delta_remove(S, x)
                
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S.remove(x)
                    current_score += delta
                    not_in_S.add(x)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        
            else:
                # Swap: remove x, add y
                if not S or not not_in_S:
                    continue
                x = random.choice(list(S))
                y = random.choice(list(not_in_S))
                
                if not is_swap_valid(S, x, y):
                    continue
                
                # Compute delta
                remove_delta = -delta_remove(S, x)
                add_delta = 0
                for o1, o2 in elem_3ap_others[y]:
                    if o1 == x or o2 == x:
                        continue
                    if o1 in S and o2 in S:
                        add_delta += 1
                delta = remove_delta + add_delta
                
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S.remove(x)
                    S.add(y)
                    current_score += delta
                    not_in_S.remove(y)
                    not_in_S.add(x)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
            
            T *= cooling_rate
            if T < T_end:
                T = T_end
        
        return best_S, best_score
    
    # Run multiple restarts with different initial solutions
    best_S = set()
    best_score = -1
    
    # First, try greedy + local search
    for i in range(5):
        if time.time() - start_time > deadline - 20:
            break
        
        S = greedy_init(seed=i)
        S, score = local_search(S, max_time=5.0)
        
        if score > best_score:
            best_score = score
            best_S = set(S)
    
    # Then, try simulated annealing from best solution and from new random starts
    S, score = simulated_annealing(best_S, max_time=15.0)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    # More restarts with SA
    for i in range(3):
        if time.time() - start_time > deadline - 5:
            break
        
        S = greedy_init(seed=100+i)
        S, score = simulated_annealing(S, max_time=15.0)
        
        if score > best_score:
            best_score = score
            best_S = set(S)
    
    # Final local search polish
    S, score = local_search(best_S, max_time=3.0)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    return sorted(best_S)
