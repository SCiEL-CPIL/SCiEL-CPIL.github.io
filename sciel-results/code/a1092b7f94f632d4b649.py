def construct():
    import time
    import random
    import math
    
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

    # Greedy initialization: pick highest weight non-conflicting elements
    S = set()
    S_mask = 0
    for a in range(N, 0, -1):
        if (conflict[a] & S_mask) == 0:
            S_mask |= (1 << a)
            S.add(a)

    def score():
        return sum(w[a] for a in S)

    best_score = score()
    best_S = S.copy()
    best_S_mask = S_mask

    # Local search with simulated annealing
    current_score = best_score
    T = 0.1  # initial temperature
    T_min = 1e-6
    cooling = 0.9995
    max_iter = 50000
    
    random.seed(42)
    
    for iteration in range(max_iter):
        if time.time() >= deadline - 5.0:
            break
        if T < T_min:
            break
            
        # Random move: add, remove, or swap
        move_type = random.random()
        new_S = S.copy()
        new_S_mask = S_mask
        delta = 0.0
        
        if move_type < 0.3:
            # Try to add a random non-conflicting element
            candidates = [a for a in range(1, N + 1) if not (S_mask >> a) & 1 and (conflict[a] & S_mask) == 0]
            if candidates:
                a = random.choice(candidates)
                new_S.add(a)
                new_S_mask |= (1 << a)
                delta = w[a]
        elif move_type < 0.5:
            # Try to remove a random element
            if S:
                x = random.choice(list(S))
                new_S.discard(x)
                new_S_mask ^= (1 << x)
                delta = -w[x]
        else:
            # Try to swap: remove one, add another
            if S:
                x = random.choice(list(S))
                rest = S_mask ^ (1 << x)
                candidates = [a for a in range(1, N + 1) if not (S_mask >> a) & 1 and (conflict[a] & rest) == 0]
                if candidates:
                    a = random.choice(candidates)
                    new_S.discard(x)
                    new_S.add(a)
                    new_S_mask = rest | (1 << a)
                    delta = w[a] - w[x]
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            S = new_S
            S_mask = new_S_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_S = S.copy()
                best_S_mask = S_mask
        
        T *= cooling
    
    # Final greedy polish: try to add any remaining non-conflicting elements
    final_S = best_S.copy()
    final_S_mask = best_S_mask
    for a in range(N, 0, -1):
        if not (final_S_mask >> a) & 1 and (conflict[a] & final_S_mask) == 0:
            final_S_mask |= (1 << a)
            final_S.add(a)
    
    # Verify and return
    if score() > best_score:
        best_S = final_S
    
    return list(best_S)
