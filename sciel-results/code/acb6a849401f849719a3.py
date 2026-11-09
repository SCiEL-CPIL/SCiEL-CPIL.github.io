import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 256
    
    # Precompute 3-APs and 4-APs
    # 3-APs: (a, b, c) with a+d=b, b+d=c
    # 4-APs: (a, b, c, d) with a+d=b, b+d=c, c+d=d
    
    # For incremental scoring:
    # For each x, store the pairs (y, z) such that {x, y, z} forms a 3-AP.
    # Note: A 3-AP is counted once. If we add x, we gain 1 for each (y,z) in A.
    # We need to be careful not to double count if we iterate differently, but 
    # "gain when adding x" is well-defined: it's the number of 3-APs completed by x.
    
    triples = [[] for _ in range(N + 1)]
    all_3aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # The set is {a, b, c}
            # If we add a, it completes (b, c) if they are present? No, (a, b, c) is the AP.
            # The AP is defined by the set {a, b, c}.
            # If we add a, and b, c are in A, we form the AP.
            # If we add b, and a, c are in A, we form the AP.
            # If we add c, and a, b are in A, we form the AP.
            # So for each element, we list the other two.
            triples[a].append((b, c))
            triples[b].append((a, c))
            triples[c].append((a, b))
            all_3aps.append((a, b, c))
            
    # For 4-AP check:
    # We need to ensure no 4-AP is fully contained in A.
    # Precompute 4-AP masks for fast bitwise check.
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = 0
            mask |= (1 << a)
            mask |= (1 << b)
            mask |= (1 << c)
            mask |= (1 << e)
            all_4aps.append(mask)
            
    def is_4ap_free(mask):
        for ap_mask in all_4aps:
            if (mask & ap_mask) == ap_mask:
                return False
        return True
        
    def count_3aps(mask):
        # This is slow, used only for initial validation or restarts
        cnt = 0
        # We can use the triples structure? No, that's for incremental.
        # For full count, iterate all 3-APs.
        for a, b, c in all_3aps:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                cnt += 1
        return cnt
        
    def get_gain(x, mask):
        # Calculate gain in 3-APs if we add x to current set mask
        gain = 0
        for y, z in triples[x]:
            if (mask & (1 << y)) and (mask & (1 << z)):
                gain += 1
        return gain

    # Initialize with a random valid set or greedy
    # Greedy: Add elements that maximize gain without creating 4-AP
    # Or just start empty and let SA do the work? Starting empty is safe but slow to build.
    # Let's do a fast greedy initialization.
    
    current_mask = 0
    current_score = 0
    available = [i for i in range(1, N + 1)]
    random.shuffle(available)
    
    # Greedy fill
    for x in available:
        if is_4ap_free(current_mask | (1 << x)):
            g = get_gain(x, current_mask)
            # We accept if it doesn't break 4-AP constraint. 
            # To maximize, we might want to be selective, but for init, any valid set is fine.
            # Actually, let's just add all that don't break 4-AP? 
            # This might create a dense set with low 3-APs? 
            # No, adding elements generally increases 3-APs.
            # But it might block future high-gain elements?
            # For simplicity, just add if valid.
            current_mask |= (1 << x)
            current_score += g
            
    best_mask = current_mask
    best_score = current_score
    
    # Simulated Annealing
    T = 5.0
    alpha = 0.9995 # Cooling factor per step
    min_T = 0.01
    
    steps = 0
    while time.time() < deadline and T > min_T:
        # Pick a random move: Add, Remove, or Swap
        # We need to know which elements are in A and which are not.
        # N=256, so we can iterate or pick random.
        
        # To make moves efficient, we can pick a random element x.
        # If x in A, try Remove. Else try Add.
        # Or pick x in A and y not in A, try Swap.
        
        move_type = random.random()
        if move_type < 0.3:
            # Add
            # Pick random x not in A
            # Efficiently picking random not in A:
            # Just pick random 1..N, if in A, retry or pick another?
            # With N=256, retries are cheap.
            x = random.randint(1, N)
            if not (current_mask & (1 << x)):
                new_mask = current_mask | (1 << x)
                if is_4ap_free(new_mask):
                    delta = get_gain(x, current_mask)
                    new_score = current_score + delta
                    if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
        elif move_type < 0.6:
            # Remove
            x = random.randint(1, N)
            if (current_mask & (1 << x)):
                new_mask = current_mask & ~(1 << x)
                # Removing always keeps 4-AP free
                # Delta is negative of gain x would have provided?
                # Gain was calculated when x was added. 
                # Score change = - (number of 3-APs containing x that are currently present in new_mask)
                # Note: new_mask does not contain x.
                # The 3-APs lost are those (y,z) in triples[x] such that y,z in new_mask.
                loss = 0
                for y, z in triples[x]:
                    if (new_mask & (1 << y)) and (new_mask & (1 << z)):
                        loss += 1
                delta = -loss
                new_score = current_score + delta
                if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
        else:
            # Swap
            x = random.randint(1, N)
            y = random.randint(1, N)
            if x == y: continue
            in_x = bool(current_mask & (1 << x))
            in_y = bool(current_mask & (1 << y))
            
            # We want one in, one out
            if in_x == in_y:
                continue
                
            if in_x:
                # Remove x, Add y
                # Check if adding y creates 4-AP in (mask without x)
                temp_mask = current_mask & ~(1 << x)
                new_mask = temp_mask | (1 << y)
                if is_4ap_free(new_mask):
                    # Calculate score change
                    # Loss from removing x: - (3-APs with x in temp_mask)
                    # Gain from adding y: + (3-APs with y in temp_mask)
                    # Note: 3-APs involving both x and y?
                    # If a 3-AP contains both x and y, it was counted in score.
                    # When we remove x, it is lost. When we add y, it might be regained?
                    # But y is already in the set? No, y is not in current_mask.
                    # So no 3-AP contains both x and y in current state.
                    # So the sets of 3-APs affected are disjoint?
                    # 3-APs containing x: {x, a, b}.
                    # 3-APs containing y: {y, c, d}.
                    # Since x!=y and neither is in the other's set context (y not in A, x in A),
                    # they are distinct APs.
                    # Loss:
                    loss = 0
                    for a, b in triples[x]:
                        if (temp_mask & (1 << a)) and (temp_mask & (1 << b)):
                            loss += 1
                    # Gain:
                    gain = 0
                    for c, d in triples[y]:
                        if (temp_mask & (1 << c)) and (temp_mask & (1 << d)):
                            gain += 1
                    delta = gain - loss
                    new_score = current_score + delta
                    if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            else:
                # Add x, Remove y
                temp_mask = current_mask & ~(1 << y)
                new_mask = temp_mask | (1 << x)
                if is_4ap_free(new_mask):
                    loss = 0
                    for a, b in triples[y]:
                        if (temp_mask & (1 << a)) and (temp_mask & (1 << b)):
                            loss += 1
                    gain = 0
                    for c, d in triples[x]:
                        if (temp_mask & (1 << c)) and (temp_mask & (1 << d)):
                            gain += 1
                    delta = gain - loss
                    new_score = current_score + delta
                    if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                            
        T *= alpha
        steps += 1
        if steps % 10000 == 0:
            # Optional: restart if stuck?
            # For now, just continue cooling.
            pass

    # Convert best_mask to sorted list
    result = [i for i in range(1, N + 1) if (best_mask & (1 << i))]
    return result
