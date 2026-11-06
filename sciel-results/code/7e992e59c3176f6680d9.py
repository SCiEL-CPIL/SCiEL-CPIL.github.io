import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    # 4-AP: (a, a+d, a+2d, a+3d)
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map elements to indices of APs they are part of
    # For faster incremental updates
    elem_to_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # Helper to count 3-APs in a set (used for initialization and final verification)
    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s:
                count += 1
        return count
        
    def has_4ap(s):
        for ap in four_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s and ap[3] in s:
                return True
        return False

    def build_base_set():
        """Build the classic 4-AP-free set: numbers with no '1' in base-3."""
        s = set()
        for x in range(1, N + 1):
            valid = True
            n = x
            while n > 0:
                if n % 3 == 1:
                    valid = False
                    break
                n //= 3
            if valid:
                s.add(x)
        return s

    # State: set of elements
    # We use a set for membership checks, but for SA we need to iterate elements.
    # Let's use a list for elements in the set to pick random x, and a set for O(1) lookup.
    
    def get_initial_set():
        return build_base_set()
    
    s = get_initial_set()
    best_s = set(s)
    best_score = count_3aps(best_s)
    
    # Simulated Annealing
    # State variables
    current_score = best_score
    current_elements = list(s)
    current_set = set(s)
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    # We will run for a fixed number of steps or until time runs out
    # Estimate steps: ~100k-500k steps per second? 
    # Incremental update is fast. Let's aim for as many as possible.
    
    # Precompute neighbors for speed? No, random swap is better for SA.
    
    # To speed up random selection, keep a list of elements in s and list of elements not in s.
    # But maintaining two lists with swaps is tricky.
    # Alternative: Pick random x from 1..N, if in s, pick random y from 1..N, if not in s.
    # This is O(1) average if density is moderate.
    
    # Let's refine the swap logic for speed.
    # Delta calculation:
    # Removing x:
    #   For each 3-AP containing x, check if other 2 are in current_set.
    # Adding y:
    #   For each 3-AP containing y, check if other 2 are in current_set \ {x}.
    #   Note: if x is one of the other 2, it's not in current_set \ {x}.
    
    # 4-AP check for adding y to current_set \ {x}:
    #   For each 4-AP containing y, check if other 3 are in current_set \ {x}.
    
    # Optimization:
    # Precompute for each element x, the list of 3-APs.
    # For a given AP (a,b,c), if we remove x, we lose 1 if the other two are in S.
    
    # Let's implement the core loop.
    
    # To make random selection efficient, we can just pick random indices from 1 to N.
    # If x in s and y not in s, proceed.
    
    # We need to track current_score incrementally.
    
    # Initial setup for SA
    # Let's run multiple restarts if time permits?
    # Or one long run with cooling.
    
    # Cooling: T = T_start * (T_end/T_start)^(iter/max_iter)
    # We don't know max_iter. Let's use time-based cooling or fixed steps.
    # Let's do 200,000 steps.
    
    max_steps = 200000
    for i in range(max_steps):
        if time.time() > deadline - 5:
            break
            
        # Cooling
        # Linear decay in log T? Or exponential.
        # T = T_start * (T_end / T_start) ** (i / max_steps)
        T = T_start * (T_end / T_start) ** (i / max_steps)
        
        # Pick random x in s, y not in s
        # To avoid infinite loop on bad luck, we can try a few times
        # But with N=256, probability is high.
        
        # Efficient random pick:
        # If we maintain a list `in_s` and `not_in_s`, picking is O(1).
        # Let's maintain these lists.
        
        # Wait, the current code doesn't maintain them. Let's initialize them.
        # This requires re-initializing if we do restarts.
        
        # Let's stick to simple random picking from 1..N for simplicity and robustness.
        # If x not in s or y in s, continue (waste a step, but rare).
        
        x = random.randint(1, N)
        if x not in current_set:
            continue
        y = random.randint(1, N)
        if y in current_set:
            continue
        
        # Check validity of adding y to current_set \ {x}
        # Check 4-APs containing y
        valid = True
        for ap_idx in elem_to_4aps[y]:
            a, b, c, d = four_aps[ap_idx]
            # Get the other 3 elements
            others = []
            if y == a: others = [b, c, d]
            elif y == b: others = [a, c, d]
            elif y == c: others = [a, b, d]
            else: others = [a, b, c]
            
            # Check if all others are in current_set AND none is x
            # Actually, if x is in others, then x is not in current_set \ {x}.
            # So we just check if all others are in current_set and x is NOT in others?
            # No. The set is S' = S \ {x} \cup {y}.
            # We check if 4-AP is formed in S'.
            # The 4-AP contains y. The other 3 elements must be in S \ {x}.
            # So for each z in others:
            #   z must be in S.
            #   z must not be x.
            
            # Since x is in S, if z == x, then z is not in S \ {x}.
            # So condition: all(z in current_set and z != x for z in others)
            
            if all(z in current_set and z != x for z in others):
                valid = False
                break
                
        if not valid:
            continue
            
        # Calculate Delta in 3-APs
        # Loss from removing x
        loss = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            # Other two elements
            if x == a:
                others = [b, c]
            elif x == b:
                others = [a, c]
            else:
                others = [a, b]
                
            # If both others are in current_set, we lose this 3-AP
            if others[0] in current_set and others[1] in current_set:
                loss += 1
                
        # Gain from adding y
        # Note: The set is S \ {x} \cup {y}.
        # We check 3-APs containing y.
        # The other two elements must be in S \ {x}.
        gain = 0
        for ap_idx in elem_to_3aps[y]:
            a, b, c = three_aps[ap_idx]
            if y == a:
                others = [b, c]
            elif y == b:
                others = [a, c]
            else:
                others = [a, b]
                
            # If both others are in current_set AND neither is x
            if (others[0] in current_set and others[0] != x) and (others[1] in current_set and others[1] != x):
                gain += 1
                
        delta = gain - loss
        
        # Acceptance probability
        if delta > 0 or random.random() < math.exp(delta / T):
            # Perform swap
            current_set.remove(x)
            current_set.add(y)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_s = set(current_set)
                
    # Final verification (optional, but good for safety)
    # If best_s has a 4-AP, it's invalid. 
    # Our logic ensures validity.
    
    return sorted(best_s)
