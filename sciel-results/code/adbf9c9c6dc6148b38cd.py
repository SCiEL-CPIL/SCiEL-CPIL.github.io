import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    # Precompute all 3-APs and 4-APs
    three_aps = []
    four_aps = []
    
    # Map element to list of 3-AP indices
    elem_to_3ap = [[] for _ in range(N+1)]
    # Map element to list of 4-AP indices
    elem_to_4ap = [[] for _ in range(N+1)]
    
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            idx = len(three_aps)
            three_aps.append((a, b, c))
            elem_to_3ap[a].append(idx)
            elem_to_3ap[b].append(idx)
            elem_to_3ap[c].append(idx)
    
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            idx = len(four_aps)
            four_aps.append((a, b, c, e))
            elem_to_4ap[a].append(idx)
            elem_to_4ap[b].append(idx)
            elem_to_4ap[c].append(idx)
            elem_to_4ap[e].append(idx)
    
    num_3aps = len(three_aps)
    num_4aps = len(four_aps)
    
    # Convert to numpy arrays for vectorized operations
    three_aps_arr = np.array(three_aps, dtype=np.int32)
    four_aps_arr = np.array(four_aps, dtype=np.int32)
    
    # Precompute: for each element, the list of other elements in 4-APs with it
    # conflict_4ap[x] = list of (other_elements_tuple, ap_index) 
    # Actually, let's precompute for each 4-AP the set of elements as a bitmask
    # But 256 bits is too large for a single int in Python... use two 128-bit or just use sets
    
    # For fast 4-AP checking: for element x, precompute all 4-APs containing x
    # and for each, store the other 3 elements
    x_to_4ap_others = [[] for _ in range(N+1)]
    for idx, (a, b, c, e) in enumerate(four_aps):
        x_to_4ap_others[a].append((b, c, e))
        x_to_4ap_others[b].append((a, c, e))
        x_to_4ap_others[c].append((a, b, e))
        x_to_4ap_others[e].append((a, b, c))
    
    # For fast 3-AP counting: for each element, list of (other1, other2, ap_index)
    x_to_3ap_others = [[] for _ in range(N+1)]
    for idx, (a, b, c) in enumerate(three_aps):
        x_to_3ap_others[a].append((b, c))
        x_to_3ap_others[b].append((a, c))
        x_to_3ap_others[c].append((a, b))
    
    def creates_4ap_fast(x, S):
        """Check if adding x to S creates a 4-AP. S is a set."""
        for others in x_to_4ap_others[x]:
            if others[0] in S and others[1] in S and others[2] in S:
                return True
        return False
    
    def count_3aps(S):
        """Count 3-APs in set S."""
        count = 0
        for a, b, c in three_aps:
            if a in S and b in S and c in S:
                count += 1
        return count
    
    def delta_3ap_add(x, S):
        """Number of new 3-APs created when adding x to S."""
        delta = 0
        for o1, o2 in x_to_3ap_others[x]:
            if o1 in S and o2 in S:
                delta += 1
        return delta
    
    def delta_3ap_remove(x, S):
        """Number of 3-APs lost when removing x from S."""
        delta = 0
        for o1, o2 in x_to_3ap_others[x]:
            if o1 in S and o2 in S:
                delta += 1
        return delta
    
    # Potential: number of 3-APs each element participates in
    potential = [len(x_to_3ap_others[i]) for i in range(N+1)]
    
    best_score = -1
    best_set = set()
    
    # Strategy: Multiple greedy inits with different orderings, then intensive local search
    
    # Greedy ordering strategies
    def greedy_init(order):
        S = set()
        for x in order:
            if not creates_4ap_fast(x, S):
                S.add(x)
        return S
    
    # Local search: add, remove, swap moves
    def local_search(S, max_iter=5000):
        S = S.copy()
        score = count_3aps(S)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if time.time() - start_time > deadline:
                break
            
            # Try additions
            best_add = None
            best_add_delta = 0
            for x in range(1, N+1):
                if x not in S:
                    if not creates_4ap_fast(x, S):
                        d = delta_3ap_add(x, S)
                        if d > best_add_delta:
                            best_add_delta = d
                            best_add = x
            
            if best_add is not None:
                S.add(best_add)
                score += best_add_delta
                improved = True
                continue
            
            # Try swaps: remove x, add y
            best_swap = None
            best_swap_delta = 0
            S_list = list(S)
            for x in S_list:
                # For each x in S, try swapping with random y not in S
                for _ in range(5):
                    y = random.randint(1, N)
                    if y in S:
                        continue
                    
                    # Check if y can be added to S\{x}
                    valid = True
                    for others in x_to_4ap_others[y]:
                        o1, o2, o3 = others
                        # Check if all of o1,o2,o3 are in S and none is x
                        # If x is among them, it's fine (removed)
                        if x in (o1, o2, o3):
                            # x is being removed, so this 4-AP is broken
                            continue
                        if o1 in S and o2 in S and o3 in S:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate delta
                    rem_delta = delta_3ap_remove(x, S)
                    # add_delta: 3-APs with y where other two are in S\{x}
                    add_delta = 0
                    for o1, o2 in x_to_3ap_others[y]:
                        if o1 == x or o2 == x:
                            continue
                        if o1 in S and o2 in S:
                            add_delta += 1
                    
                    total_delta = add_delta - rem_delta
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
            
            # Try random removals to escape local optima
            if len(S) > 10:
                x = random.choice(list(S))
                S.remove(x)
                score -= delta_3ap_remove(x, S)
                improved = True
                continue
        
        return S, score
    
    # Simulated annealing
    def simulated_annealing(S, iterations=20000):
        S = S.copy()
        score = count_3aps(S)
        T = 5.0
        cooling = 0.9995
        
        for _ in range(iterations):
            if time.time() - start_time > deadline:
                break
            
            # Random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                candidates = [x for x in range(1, N+1) if x not in S]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if creates_4ap_fast(x, S):
                    continue
                delta = delta_3ap_add(x, S)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S.add(x)
                    score += delta
            
            elif move_type == 'remove':
                if len(S) <= 5:
                    continue
                x = random.choice(list(S))
                delta = -delta_3ap_remove(x, S)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S.remove(x)
                    score += delta
            
            elif move_type == 'swap':
                if len(S) < 5:
                    continue
                x = random.choice(list(S))
                y = random.randint(1, N)
                if y in S:
                    continue
                
                # Check validity of swap
                valid = True
                for others in x_to_4ap_others[y]:
                    o1, o2, o3 = others
                    if x in (o1, o2, o3):
                        continue
                    if o1 in S and o2 in S and o3 in S:
                        valid = False
                        break
                if not valid:
                    continue
                
                rem_delta = delta_3ap_remove(x, S)
                add_delta = 0
                for o1, o2 in x_to_3ap_others[y]:
                    if o1 == x or o2 == x:
                        continue
                    if o1 in S and o2 in S:
                        add_delta += 1
                
                delta = add_delta - rem_delta
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S.remove(x)
                    S.add(y)
                    score += delta
            
            T *= cooling
        
        return S, score
    
    # Run multiple restarts with different strategies
    for restart in range(30):
        if time.time() - start_time > deadline:
            break
        
        # Different greedy orderings
        if restart % 3 == 0:
            # Sort by potential descending
            order = sorted(range(1, N+1), key=lambda x: -potential[x])
        elif restart % 3 == 1:
            # Random order
            order = list(range(1, N+1))
            random.shuffle(order)
        else:
            # Sort by potential ascending (try different structure)
            order = sorted(range(1, N+1), key=lambda x: potential[x])
        
        S = greedy_init(order)
        score = count_3aps(S)
        
        if score > best_score:
            best_score = score
            best_set = S.copy()
        
        # Local search
        S, score = local_search(S)
        if score > best_score:
            best_score = score
            best_set = S.copy()
        
        # Simulated annealing on the best found
        if time.time() - start_time < deadline - 10:
            S_sa, score_sa = simulated_annealing(best_set)
            if score_sa > best_score:
                best_score = score_sa
                best_set = S_sa.copy()
        
        # Final local search polish
        if time.time() - start_time < deadline - 5:
            S, score = local_search(best_set, max_iter=3000)
            if score > best_score:
                best_score = score
                best_set = S.copy()
    
    # Final verification and return
    if not best_set:
        # Fallback: just take odd numbers (no 4-AP, some 3-APs)
        best_set = set(range(1, N+1, 2))
    
    return sorted(best_set)
