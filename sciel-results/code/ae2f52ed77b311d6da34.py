"""Combined bitset local search with simulated annealing for max no-two-other-product-divisibility set."""
import time
import random

N = 256
CANDS = list(range(2, N + 1))
M = len(CANDS)
IDX = {v: i for i, v in enumerate(CANDS)}
BIT = [1 << i for i in range(M)]

def _precompute():
    """Precompute _DIV_BY[a][b] = mask of c such that CANDS[a] | CANDS[b]*CANDS[c]."""
    div_by = [[0] * M for _ in range(M)]
    for a in range(M):
        va = CANDS[a]
        row = div_by[a]
        for b in range(M):
            vb = CANDS[b]
            mask = 0
            for c in range(M):
                if (vb * CANDS[c]) % va == 0:
                    mask |= BIT[c]
            row[b] = mask
    return div_by

_DIV_BY = _precompute()

def _popcount(S):
    return bin(S).count('1')

def _members(S):
    return [CANDS[i] for i in range(M) if S & BIT[i]]

def _violation_count(S):
    """Count the number of violated triples in set S.
    A triple (a, b, c) distinct is violated if a | b*c.
    We count each ordered pair (a, b) in S, then count how many c in S (c != a, c != b) satisfy a | b*c.
    Each unordered triple {a,b,c} can be violated in up to 3 ways (a|b*c, b|a*c, c|a*b),
    but we just need the total count for energy purposes.
    """
    total = 0
    mems = [i for i in range(M) if S & BIT[i]]
    for a in mems:
        ba = BIT[a]
        row_a = _DIV_BY[a]
        for b in mems:
            if b == a:
                continue
            # count c in S, c != a, c != b, such that a | b*c
            mask = row_a[b] & S & ~ba & ~BIT[b]
            total += bin(mask).count('1')
    return total

def _can_add(S, x):
    """Check if x can be added to set S without creating any new violations."""
    if S & BIT[x]:
        return False
    bx = BIT[x]
    dx = _DIV_BY[x]
    for a in range(M):
        ba = BIT[a]
        if not (S & ba):
            continue
        # triple (x, a, b): x | a*b for some b in S, b != x, b != a
        if (dx[a] & S & ~ba & ~bx):
            return False
        # triple (a, x, b): a | x*b for some b in S, b != a, b != x
        if (_DIV_BY[a][x] & S & ~ba & ~bx):
            return False
    return True

def _local_add(S):
    """Repeatedly add any element that can be added."""
    changed = True
    while changed:
        changed = False
        for x in range(M):
            if not (S & BIT[x]) and _can_add(S, x):
                S |= BIT[x]
                changed = True
                break
    return S

def _greedy(order):
    S = 0
    for x in order:
        if _can_add(S, x):
            S |= BIT[x]
    return S

def _swap_improve(S, rng, rounds=5):
    """Try 1-1 swaps that keep size but may enable additions."""
    for _ in range(rounds):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        best_add = None
        for x in range(M):
            if x == y or (S1 & BIT[x]):
                continue
            if _can_add(S1, x):
                best_add = x
                break
        if best_add is not None:
            S2 = S1 | BIT[best_add]
            S2 = _local_add(S2)
            if _popcount(S2) >= _popcount(S):
                S = S2
    return S

def _two_exchange(S, rng, trials=50):
    """Remove one, add two."""
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        cands = []
        for x1 in range(M):
            if x1 == y or (S1 & BIT[x1]):
                continue
            if _can_add(S1, x1):
                cands.append(x1)
        rng.shuffle(cands)
        for x1 in cands[:3]:
            S2 = S1 | BIT[x1]
            for x2 in range(M):
                if x2 == y or x2 == x1 or (S2 & BIT[x2]):
                    continue
                if _can_add(S2, x2):
                    S3 = S2 | BIT[x2]
                    S3 = _local_add(S3)
                    if _popcount(S3) > best_sz:
                        best = S3
                        best_sz = _popcount(S3)
                    break
    return best

def _simulated_annealing(S, rng, deadline):
    """Simulated annealing to minimize violations, tracking best valid set."""
    best_S = S
    best_size = _popcount(S)
    best_violations = _violation_count(S)
    
    if best_violations == 0:
        best_S = _local_add(best_S)
        best_size = _popcount(best_S)
        return best_S, best_size
    
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    current_violations = best_violations
    current_S = S
    
    while T > T_min and time.time() < deadline:
        # Choose a random flip: add or remove a random element
        mems = [i for i in range(M) if current_S & BIT[i]]
        non_mems = [i for i in range(M) if not (current_S & BIT[i])]
        
        if rng.random() < 0.5 and non_mems:
            # Add a random element
            x = rng.choice(non_mems)
            new_S = current_S | BIT[x]
        elif mems and len(mems) > 1:
            # Remove a random element
            x = rng.choice(mems)
            new_S = current_S & ~BIT[x]
        else:
            break
        
        # Compute new violations
        new_violations = _violation_count(new_S)
        delta = new_violations - current_violations
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            current_S = new_S
            current_violations = new_violations
            
            # Track best valid set
            if current_violations == 0:
                # Try to add more
                polished = _local_add(current_S)
                sz = _popcount(polished)
                if sz > best_size:
                    best_size = sz
                    best_S = polished
                    current_S = polished
                    current_violations = 0
        
        T *= cooling
    
    # Final polish
    if _violation_count(current_S) == 0:
        polished = _local_add(current_S)
        sz = _popcount(polished)
        if sz > best_size:
            best_size = sz
            best_S = polished
    
    return best_S, best_size

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin before 90s limit
    
    # Baseline: primes
    prime_S = 0
    for v in CANDS:
        if _is_prime(v):
            prime_S |= BIT[IDX[v]]
    best_S = prime_S
    best_size = _popcount(prime_S)
    
    # Local add on primes
    best_S = _local_add(best_S)
    best_size = _popcount(best_S)
    
    # Multiple restarts with different strategies
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        
        # Strategy 1: Random greedy
        if seed % 3 == 0:
            order = list(range(M))
            rng.shuffle(order)
            S = _greedy(order)
            S = _local_add(S)
            sz = _popcount(S)
            if sz > best_size:
                best_size = sz
                best_S = S
        
        # Strategy 2: Perturb prime set, then local search
        elif seed % 3 == 1:
            P = prime_S
            mems = [i for i in range(M) if P & BIT[i]]
            k = rng.randint(1, 5)
            for _ in range(k):
                if not mems:
                    break
                y = rng.choice(mems)
                P &= ~BIT[y]
                mems = [i for i in range(M) if P & BIT[i]]
            P = _local_add(P)
            P = _swap_improve(P, rng)
            P = _two_exchange(P, rng)
            P = _local_add(P)
            sz = _popcount(P)
            if sz > best_size:
                best_size = sz
                best_S = P
        
        # Strategy 3: Simulated annealing from a perturbed starting point
        else:
            # Start from a random subset or perturbed prime set
            if seed % 6 == 2:
                # Start from a random subset of size ~best_size
                all_mems = list(range(M))
                rng.shuffle(all_mems)
                S = 0
                target_size = max(20, best_size - rng.randint(5, 15))
                for i in range(min(target_size, M)):
                    S |= BIT[all_mems[i]]
            else:
                # Start from perturbed prime set
                P = prime_S
                mems = [i for i in range(M) if P & BIT[i]]
                k = rng.randint(2, 8)
                for _ in range(k):
                    if not mems:
                        break
                    y = rng.choice(mems)
                    P &= ~BIT[y]
                    mems = [i for i in range(M) if P & BIT[i]]
                S = P
            
            # Run simulated annealing
            S, sz = _simulated_annealing(S, rng, deadline - 3)
            if sz > best_size:
                best_size = sz
                best_S = S
    
    # Final local search on best
    best_S = _local_add(best_S)
    best_S = _swap_improve(best_S, random.Random(42), rounds=10)
    best_S = _two_exchange(best_S, random.Random(43), trials=100)
    best_S = _local_add(best_S)
    
    # Verify validity
    final_mems = _members(best_S)
    # Quick validation: check no violations
    valid = True
    for i in range(len(final_mems)):
        a = final_mems[i]
        for j in range(len(final_mems)):
            if j == i:
                continue
            b = final_mems[j]
            for k in range(len(final_mems)):
                if k == i or k == j:
                    continue
                c = final_mems[k]
                if (b * c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to primes if something went wrong
        return [int(v) for v in CANDS if _is_prime(v)]
    
    return [int(x) for x in final_mems]
