import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0

    # Precompute APs
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_indices_by_num = {i: [] for i in range(1, N + 1)}
    ap_others = []
    for idx, ap in enumerate(all_aps):
        others = []
        for i in range(4):
            o = [ap[j] for j in range(4) if j != i]
            others.append(tuple(o))
        ap_others.append(others)
        for x in ap:
            ap_indices_by_num[x].append(idx)
            
    x_ap_pos = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for pos, x in enumerate(ap):
            x_ap_pos[x].append((idx, pos))

    def can_add(x, in_set):
        for ap_idx, pos in x_ap_pos[x]:
            others = ap_others[ap_idx][pos]
            if in_set[others[0]] and in_set[others[1]] and in_set[others[2]]:
                return False
        return True

    def set_to_bool(S):
        in_set = [False] * (N + 1)
        for x in S:
            in_set[x] = True
        return in_set

    def bool_to_set(in_set):
        return {i for i in range(1, N + 1) if in_set[i]}

    # Initial Heuristics
    best_set = set()
    best_size = 0
    
    # 1. Base-4
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)
            
    if len(base4_set) > best_size:
        best_set = base4_set
        best_size = len(base4_set)
        
    # 2. Randomized Greedy Initializations
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = set()
        in_set = [False] * (N + 1)
        for x in order:
            if can_add(x, in_set):
                S.add(x)
                in_set[x] = True
        if len(S) > best_size:
            best_set = S
            best_size = len(S)

    # 3. Iterated Local Search
    # State: best_set
    # Move: Remove k random elements, greedily refill
    
    stuck_count = 0
    current_k = 1
    
    while time.time() < deadline - 2:
        # Determine k
        if stuck_count > 100:
            current_k = random.randint(3, 5)
            stuck_count = 0
        else:
            current_k = random.randint(1, 3)
            
        # Start from best
        S_curr = set(best_set)
        in_set = set_to_bool(S_curr)
        
        # Remove k random elements
        elements = list(S_curr)
        random.shuffle(elements)
        to_remove = elements[:current_k]
        
        for x in to_remove:
            S_curr.remove(x)
            in_set[x] = False
            
        # Greedy Refill
        candidates = [x for x in range(1, N + 1) if not in_set[x]]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, in_set):
                S_curr.add(x)
                in_set[x] = True
                
        new_size = len(S_curr)
        
        if new_size > best_size:
            best_set = S_curr
            best_size = new_size
            stuck_count = 0
        elif new_size == best_size:
            # Accept equal size move with some probability to drift
            if random.random() < 0.5:
                best_set = S_curr
                # Do not reset stuck_count immediately, but maybe decay it?
                # Actually, if we are stuck in size, drifting is good.
                stuck_count = max(0, stuck_count - 1)
            else:
                stuck_count += 1
        else:
            stuck_count += 1

    return sorted(best_set)
