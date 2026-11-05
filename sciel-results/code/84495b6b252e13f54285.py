import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    # Helper to check Sidon and get sums
    def get_sums(S):
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return None, False
                sums.add(s)
        return sums, True

    # Helper to check maximality and count addable elements
    def count_addable(S, sums):
        addable = 0
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            # New sums: x + a for a in S, and 2x
            conflict = False
            for a in S:
                if (x + a) in sums or (2 * x) in sums:
                    conflict = True
                    break
            # Also need to check if x+a conflicts with other x+b?
            # No, sums in S are distinct.
            # We just need to check if new sums intersect existing sums.
            # x+a and x+b are distinct if a!=b.
            # 2x and x+a: 2x = x+a => x=a. But x not in S. So distinct.
            if not conflict:
                addable += 1
        return addable

    def cost(S):
        sums, is_sidon = get_sums(S)
        if not is_sidon:
            return 10000 + len(S)  # Heavy penalty for invalid Sidon
        addable = count_addable(S, sums)
        return len(S) + 10 * addable

    # Initialize with a random Sidon set
    # Greedy random to get a valid Sidon set
    S = []
    sums = set()
    # Shuffle candidates
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        added = [x + a for a in S] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            S.append(x)
            sums.update(added)
            
    best_S = S[:]
    best_cost = cost(S)
    
    current_S = S[:]
    current_cost = best_cost
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Propose move
        move_type = random.choice(['add', 'remove', 'swap'])
        new_S = current_S[:]
        
        if move_type == 'add':
            # Pick random x not in S
            available = [x for x in range(1, N + 1) if x not in new_S]
            if available:
                x = random.choice(available)
                new_S.append(x)
        elif move_type == 'remove':
            if new_S:
                x = random.choice(new_S)
                new_S.remove(x)
        else: # swap
            if len(new_S) >= 2:
                x = random.choice(new_S)
                available = [y for y in range(1, N + 1) if y not in new_S]
                if available:
                    y = random.choice(available)
                    new_S.remove(x)
                    new_S.append(y)
            
        new_cost = cost(new_S)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_S = new_S
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = current_S[:]
                
        T *= cooling
        
    # Verify best_S is maximal Sidon
    sums, is_sidon = get_sums(best_S)
    if is_sidon:
        addable = count_addable(best_S, sums)
        if addable == 0:
            return best_S
            
    # Fallback: return the seed greedy result if SA failed to find a valid maximal set
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected
