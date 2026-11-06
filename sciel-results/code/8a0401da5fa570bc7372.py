import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    all_3aps = []
    all_4aps = []
    
    # 3-AP: (a, a+d, a+2d)
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                all_3aps.append((a, b, c))
                
    # 4-AP: (a, a+d, a+2d, a+3d)
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_4aps.append((a, b, c, e))
    
    # Index APs by element
    threes_by_elem = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(all_3aps):
        threes_by_elem[a].append(idx)
        threes_by_elem[b].append(idx)
        threes_by_elem[c].append(idx)
        
    fours_by_elem = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(all_4aps):
        fours_by_elem[a].append(idx)
        fours_by_elem[b].append(idx)
        fours_by_elem[c].append(idx)
        fours_by_elem[e].append(idx)
        
    # Helper to check if adding x creates a 4-AP given current S (bool array)
    def creates_4ap(S, x):
        for idx in fours_by_elem[x]:
            a, b, c, e = all_4aps[idx]
            # Check if other 3 are in S
            # x is one of a,b,c,e
            if x == a:
                if S[b] and S[c] and S[e]:
                    return True
            elif x == b:
                if S[a] and S[c] and S[e]:
                    return True
            elif x == c:
                if S[a] and S[b] and S[e]:
                    return True
            elif x == e:
                if S[a] and S[b] and S[c]:
                    return True
        return False
        
    # Helper to count 3-APs gained/lost by adding/removing x
    def gain_3aps(S, x):
        count = 0
        for idx in threes_by_elem[x]:
            a, b, c = all_3aps[idx]
            if x == a:
                if S[b] and S[c]:
                    count += 1
            elif x == b:
                if S[a] and S[c]:
                    count += 1
            elif x == c:
                if S[a] and S[b]:
                    count += 1
        return count

    # Initial construction: Randomized Greedy
    def initial_construct():
        S = [False] * (N + 1)
        S_set = set()
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        # Try to add elements in random order if they don't create 4-AP
        # This is a simple heuristic to get a valid set
        for x in remaining:
            if not creates_4ap(S, x):
                S[x] = True
                S_set.add(x)
                
        return S, S_set

    # Simulated Annealing
    def simulated_annealing(S_init, S_set_init, time_budget=70):
        S = S_init[:]
        S_set = set(S_set_init)
        
        # Calculate initial 3-AP count
        current_3ap = 0
        for a, b, c in all_3aps:
            if S[a] and S[b] and S[c]:
                current_3ap += 1
                
        best_S = S[:]
        best_S_set = set(S_set)
        best_3ap = current_3ap
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # Precompute list of elements for random selection
        all_elements = list(range(1, N + 1))
        
        while time.time() < start_time + time_budget and T > min_T:
            # Choose move type: 0=Add, 1=Remove, 2=Swap
            move_type = random.randint(0, 2)
            
            if move_type == 0: # Add
                # Pick random x not in S
                candidates = [x for x in all_elements if not S[x]]
                if not candidates:
                    continue
                x = random.choice(candidates)
                
                # Check 4-AP
                if creates_4ap(S, x):
                    continue
                    
                gain = gain_3aps(S, x)
                delta = gain
                
                # Acceptance
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[x] = True
                    S_set.add(x)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                        
            elif move_type == 1: # Remove
                if not S_set:
                    continue
                x = random.choice(list(S_set))
                
                loss = gain_3aps(S, x) # Same logic, it's the count of APs containing x
                delta = -loss
                
                # Acceptance
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[x] = False
                    S_set.discard(x)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                        
            else: # Swap
                if not S_set:
                    continue
                # Pick x to remove
                x = random.choice(list(S_set))
                # Pick y to add
                candidates = [y for y in all_elements if not S[y]]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                # Check if adding y creates 4-AP after removing x
                # We can check if y creates 4-AP with S \ {x}
                # This is slightly complex. 
                # Simpler: 
                # 1. Temporarily remove x
                # 2. Check if y creates 4-AP
                # 3. If ok, calculate delta
                # 4. If not ok or rejected, revert
                
                S[x] = False
                S_set.discard(x)
                
                if creates_4ap(S, y):
                    # Revert
                    S[x] = True
                    S_set.add(x)
                    T *= cooling_rate
                    continue
                
                loss_x = gain_3aps(S, x) # APs lost by removing x (calculated on S without x? No, gain_3aps uses S)
                # Wait, gain_3aps(S, x) counts APs where x and other 2 are in S.
                # If we remove x, we lose those APs.
                # So loss is gain_3aps(S_original, x).
                # But we already modified S. 
                # Let's calculate loss on S_original.
                # We didn't save S_original.
                # Let's re-calculate.
                # Actually, gain_3aps(S, x) where S does NOT contain x is the same as the number of APs that would be formed if x were added.
                # So if we have S (with x), the number of APs containing x is gain_3aps(S, x).
                # When we remove x, we lose gain_3aps(S, x).
                # When we add y to S\{x}, we gain gain_3aps(S\{x}, y).
                
                # We need to be careful.
                # Let's restore S to original state for calculation? No, that's slow.
                # Let's calculate loss_x based on S (which now has x=False).
                # The APs containing x that were valid in S_orig are exactly those where the other 2 elements are in S_orig.
                # Since x is the only change, the other 2 elements are in S (current) if and only if they are in S_orig.
                # So loss_x = gain_3aps(S, x) is correct? 
                # gain_3aps(S, x) checks if other 2 are in S. Since x is not in S, it doesn't affect the check for the other 2.
                # So yes, loss_x = gain_3aps(S, x).
                
                loss_x = gain_3aps(S, x)
                
                gain_y = gain_3aps(S, y)
                
                delta = gain_y - loss_x
                
                # Acceptance
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[y] = True
                    S_set.add(y)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                else:
                    # Revert: Put x back
                    S[x] = True
                    S_set.add(x)
                    # current_3ap is unchanged
                    
            T *= cooling_rate
            
        return best_S, best_S_set, best_3ap

    # Run multiple restarts
    best_overall_3ap = 0
    best_overall_S = []
    
    num_restarts = 5
    time_per_restart = 12
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        S_init, S_set_init = initial_construct()
        best_S, best_S_set, best_3ap = simulated_annealing(S_init, S_set_init, time_per_restart)
        
        if best_3ap > best_overall_3ap:
            best_overall_3ap = best_3ap
            best_overall_S = best_S_set
            
    # If no valid set found (should not happen), return {1}
    if not best_overall_S:
        return [1]
        
    return sorted(best_overall_S)
