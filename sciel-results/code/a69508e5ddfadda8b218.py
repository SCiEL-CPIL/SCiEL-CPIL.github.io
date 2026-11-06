def construct():
    import time
    import random
    import math
    from functools import lru_cache
    
    deadline = time.time() + 110.0
    N = 511

    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    pindex = {p: i for i, p in enumerate(primes)}

    mult_mask = {}
    for p in primes:
        m = 0
        for b in range(p, N + 1, p):
            m |= (1 << b)
        mult_mask[p] = m

    pmask = [0] * (N + 1)
    conflict = [0] * (N + 1)
    for a in range(2, N + 1):
        x = a
        pm = 0
        cm = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                pm |= (1 << pindex[p])
                cm |= mult_mask[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            pm |= (1 << pindex[x])
            cm |= mult_mask[x]
        pmask[a] = pm
        conflict[a] = cm

    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512 - a)

    # Precompute all valid elements sorted by weight descending
    all_elements = list(range(1, N + 1))
    all_elements.sort(key=lambda a: -w[a])
    
    def greedy_init(seed_offset=0):
        """Greedy initialization with optional random perturbation."""
        S = set()
        S_mask = 0
        rng = random.Random(42 + seed_offset)
        
        # Try different orderings: pure greedy, or with some randomization
        elements = all_elements[:]
        if seed_offset > 0:
            # Shuffle the bottom half to get diverse starts
            half = len(elements) // 2
            elements[half:] = elements[half:]
            rng.shuffle(elements[half:])
        
        for a in elements:
            if (conflict[a] & S_mask) == 0:
                S_mask |= (1 << a)
                S.add(a)
        return S, S_mask

    def polish(S, S_mask):
        """Add any remaining non-conflicting elements in weight order."""
        for a in all_elements:
            if not (S_mask >> a) & 1 and (conflict[a] & S_mask) == 0:
                S_mask |= (1 << a)
                S.add(a)
        return S, S_mask

    def local_search(S, S_mask, current_score, rng, max_iter, deadline):
        """Simulated annealing local search."""
        T = 0.05
        T_min = 1e-7
        cooling = 0.999
        best_score = current_score
        best_S = S.copy()
        best_S_mask = S_mask
        
        for iteration in range(max_iter):
            if time.time() >= deadline - 3.0:
                break
            if T < T_min:
                break
                
            move_type = rng.random()
            delta = 0.0
            new_S_mask = S_mask
            added = None
            removed = None
            
            if move_type < 0.35:
                # Try to add a random non-conflicting element
                candidates = [a for a in all_elements if not (S_mask >> a) & 1 and (conflict[a] & S_mask) == 0]
                if candidates:
                    a = rng.choice(candidates)
                    new_S_mask |= (1 << a)
                    delta = w[a]
                    added = a
            elif move_type < 0.55:
                # Try to remove a random element
                if S:
                    x = rng.choice(list(S))
                    new_S_mask ^= (1 << x)
                    delta = -w[x]
                    removed = x
            else:
                # Try to swap: remove one, add another
                if S:
                    x = rng.choice(list(S))
                    rest = S_mask ^ (1 << x)
                    candidates = [a for a in all_elements if not (S_mask >> a) & 1 and (conflict[a] & rest) == 0]
                    if candidates:
                        a = rng.choice(candidates)
                        new_S_mask = rest | (1 << a)
                        delta = w[a] - w[x]
                        added = a
                        removed = x
            
            if added is None and removed is None:
                T *= cooling
                continue
            
            # Acceptance criterion
            if delta > 0 or rng.random() < math.exp(delta / T):
                if added is not None:
                    S.add(added)
                if removed is not None:
                    S.discard(removed)
                S_mask = new_S_mask
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
                    best_S_mask = S_mask
            
            T *= cooling
        
        return best_S, best_S_mask, best_score

    # Multi-start simulated annealing
    global_best_S = None
    global_best_S_mask = 0
    global_best_score = -1.0
    
    num_starts = 8
    rng = random.Random(12345)
    
    for start_idx in range(num_starts):
        if time.time() >= deadline - 5.0:
            break
        
        S, S_mask = greedy_init(start_idx)
        S, S_mask = polish(S, S_mask)
        current_score = sum(w[a] for a in S)
        
        if current_score > global_best_score:
            global_best_score = current_score
            global_best_S = S.copy()
            global_best_S_mask = S_mask
        
        # Run local search
        S, S_mask, score = local_search(S, S_mask, current_score, rng, 30000, deadline)
        
        if score > global_best_score:
            global_best_score = score
            global_best_S = S.copy()
            global_best_S_mask = S_mask
    
    # Final polish on the best solution found
    final_S, final_S_mask = polish(global_best_S.copy(), global_best_S_mask)
    final_score = sum(w[a] for a in final_S)
    
    if final_score > global_best_score:
        global_best_S = final_S
        global_best_S_mask = final_S_mask
        global_best_score = final_score
    
    return list(global_best_S)
