import time
import random

N = 256
DEADLINE = 55.0

def _precompute():
    """
    Precompute data structures for fast incremental updates.
    Returns:
    ap3_partners: list of lists of tuples (j, k) for each i in 1..N
    ap4_partners: list of lists of tuples (j, k, l) for each i in 1..N
    """
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Update partners for a, b, c
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Update partners for a, b, c, e
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

def _count_3aps_slow(in_set, ap3_partners):
    """Slow count for verification or initial setup."""
    count = 0
    # Count each 3-AP once. 
    # A 3-AP is (a, b, c) with a < b < c. 
    # We can iterate a, b, c or use the partners structure.
    # Using partners: each 3-AP is counted 3 times (once for each element).
    # So sum of valid pairs / 3.
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    # Multi-restart Simulated Annealing
    # We will run several restarts. Each restart initializes randomly and runs SA.
    # Time budget per restart: ~ (DEADLINE - 5) / num_restarts
    
    num_restarts = 5
    time_per_restart = (DEADLINE - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        # Check global deadline
        if time.time() - start_time > DEADLINE - 2.0:
            break
            
        # Initialize
        in_set = [False] * (N + 1)
        # Start with a random subset of size ~ N/3 to N/4
        # Greedy random initialization:
        # Try to add elements if they don't create 4-APs
        # This ensures we start with a valid set.
        
        # To make it faster and denser, let's use a "greedy random" init:
        # Shuffle 1..N, try to add each if no 4-AP is formed.
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if adding x creates a 4-AP
            valid = True
            for j, k, l in ap4_partners[x]:
                if in_set[j] and in_set[k] and in_set[l]:
                    valid = False
                    break
            if valid:
                in_set[x] = True
        
        # Calculate initial score
        score = _count_3aps_slow(in_set, ap3_partners)
        
        # Simulated Annealing parameters
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        steps = 0
        max_steps = 20000 # Adjust based on speed
        
        # We will run until time slice is up or steps exhausted
        while steps < max_steps:
            if time.time() - start_time > DEADLINE - 1.0:
                break
            
            # Choose move type
            # 50% Swap, 25% Add, 25% Remove
            r = random.random()
            
            if r < 0.5:
                # Swap: remove x, add y
                # Pick random x in set, random y not in set
                # To speed up, maintain lists of present/absent? 
                # With N=256, scanning is cheap.
                present = [i for i in range(1, N + 1) if in_set[i]]
                absent = [i for i in range(1, N + 1) if not in_set[i]]
                
                if not present or not absent:
                    T *= cooling
                    steps += 1
                    continue
                
                x = random.choice(present)
                y = random.choice(absent)
                
                # Check 4-AP constraint for adding y to S \ {x}
                valid = True
                for j, k, l in ap4_partners[y]:
                    # Check if j, k, l are in S and not x
                    if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
                        valid = False
                        break
                
                if not valid:
                    T *= cooling
                    steps += 1
                    continue
                
                # Calculate Delta
                # Delta_remove_x: number of 3-APs containing x with other 2 in S
                delta_rem = 0
                for j, k in ap3_partners[x]:
                    if in_set[j] and in_set[k]:
                        delta_rem += 1
                        
                # Delta_add_y: number of 3-APs containing y with other 2 in S \ {x}
                delta_add = 0
                for j, k in ap3_partners[y]:
                    if (in_set[j] and j != x) and (in_set[k] and k != x):
                        delta_add += 1
                        
                delta = delta_add - delta_rem
                
                # Metropolis criterion
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    # Accept
                    in_set[x] = False
                    in_set[y] = True
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_set = in_set[:]
                        
            elif r < 0.75:
                # Add y
                absent = [i for i in range(1, N + 1) if not in_set[i]]
                if not absent:
                    T *= cooling
                    steps += 1
                    continue
                y = random.choice(absent)
                
                # Check 4-AP
                valid = True
                for j, k, l in ap4_partners[y]:
                    if in_set[j] and in_set[k] and in_set[l]:
                        valid = False
                        break
                if not valid:
                    T *= cooling
                    steps += 1
                    continue
                    
                # Delta
                delta = 0
                for j, k in ap3_partners[y]:
                    if in_set[j] and in_set[k]:
                        delta += 1
                        
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    in_set[y] = True
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_set = in_set[:]
                        
            else:
                # Remove x
                present = [i for i in range(1, N + 1) if in_set[i]]
                if not present:
                    T *= cooling
                    steps += 1
                    continue
                x = random.choice(present)
                
                # Delta
                delta = 0
                for j, k in ap3_partners[x]:
                    if in_set[j] and in_set[k]:
                        delta += 1
                        
                delta = -delta # Removing decreases score by this amount
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    in_set[x] = False
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_set = in_set[:]
                        
            T *= cooling
            if T < T_min:
                T = T_min
            steps += 1
            
    # Convert best_set to list
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
