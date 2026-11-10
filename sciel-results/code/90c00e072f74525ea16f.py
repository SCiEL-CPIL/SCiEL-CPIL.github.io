import random
import time

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    # Precompute 4-APs
    aps = []
    elem_to_aps = [[] for _ in range(N+1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            for x in (a, b, c, e):
                elem_to_aps[x].append(ap_idx)
                
    # Precompute "others" for faster check
    x_others = [[] for _ in range(N+1)]
    for idx, ap in enumerate(aps):
        a, b, c, e = ap
        x_others[a].append((b, c, e))
        x_others[b].append((a, c, e))
        x_others[c].append((a, b, e))
        x_others[e].append((a, b, c))

    def is_valid(x, in_S):
        for o1, o2, o3 in x_others[x]:
            if in_S[o1] and in_S[o2] and in_S[o3]:
                return False
        return True

    def greedy_fill(in_S, rng):
        order = list(range(1, N+1))
        rng.shuffle(order)
        for x in order:
            if in_S[x]: continue
            if is_valid(x, in_S):
                in_S[x] = True
        return in_S

    best_size = 0
    best_in_S = [False] * (N+1)
    rng = random.Random(42)
    
    # Main Loop
    while time.time() < deadline:
        # 1. Random Greedy Start
        in_S = [False] * (N+1)
        greedy_fill(in_S, rng)
        
        current_size = sum(in_S)
        if current_size > best_size:
            best_size = current_size
            best_in_S = in_S.copy()
            
        # 2. Local Improvement: Remove and Refill
        # Try several times
        for _ in range(50): # 50 attempts per restart
            if time.time() > deadline: break
            
            # Get list of elements in current best
            current_elements = [i for i in range(1, N+1) if best_in_S[i]]
            if not current_elements: continue
            
            u = rng.choice(current_elements)
            
            # Create a copy of best_in_S
            trial_in_S = best_in_S.copy()
            trial_in_S[u] = False
            
            # Greedy Refill
            greedy_fill(trial_in_S, rng)
            
            new_size = sum(trial_in_S)
            if new_size > best_size:
                best_size = new_size
                best_in_S = trial_in_S

    return sorted([i for i in range(1, N+1) if best_in_S[i]])
