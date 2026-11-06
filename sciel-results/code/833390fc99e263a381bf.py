def construct():
    import time
    from math import gcd
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    N = 512
    # weights: weight[a] = 1/(512 - a) for a in 1..511
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # Precompute prime factors for fast gcd check
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    factors = {a: prime_factors(a) for a in range(1, N)}
    factors[1] = set()
    
    def is_coprime(a, b):
        if a == 1 or b == 1:
            return True
        return len(factors[a] & factors[b]) == 0
    
    def calc_score(chosen):
        return sum(weights[a] for a in chosen)
    
    # Initial Greedy Construction
    # Sort candidates by weight descending (which is equivalent to a descending since weight is monotonic with a)
    candidates = list(range(1, N))
    candidates.sort(key=lambda a: -weights[a])
    
    chosen = []
    for a in candidates:
        if all(is_coprime(a, b) for b in chosen):
            chosen.append(a)
            
    best = chosen[:]
    best_score = calc_score(best)
    
    # Local Search
    # We try to improve the set by removing one element and re-greedy-ing, 
    # or by swapping one element with another.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Strategy 1: Remove one element and re-greedy
        for i, a in enumerate(best):
            if time.time() > deadline:
                break
            
            remaining = [x for x in best if x != a]
            new_chosen = remaining[:]
            
            # Re-greedy: add best available candidates
            for b in candidates:
                if b in new_chosen:
                    continue
                if all(is_coprime(b, c) for c in new_chosen):
                    new_chosen.append(b)
                    # We can stop early if the remaining potential gain is small, 
                    # but for N=512, full pass is fast enough.
            
            new_score = calc_score(new_chosen)
            if new_score > best_score + 1e-12:
                best = new_chosen
                best_score = new_score
                improved = True
                break # Restart loop with new best
        
        # Strategy 2: Swap one element with another
        # If we didn't improve in the previous loop, try swaps
        if not improved and time.time() < deadline:
            for i, a in enumerate(best):
                if time.time() > deadline:
                    break
                # Try to swap 'a' with 'b'
                # 'b' must be coprime to all other elements in best (excluding a)
                others = [x for x in best if x != a]
                # We want b with weight > weight[a]
                # Iterate candidates in descending weight
                for b in candidates:
                    if b == a or b in best:
                        continue
                    if weights[b] <= weights[a]:
                        break # No point checking smaller weights if we want improvement
                    if all(is_coprime(b, c) for c in others):
                        # Swap is valid and improves score
                        new_chosen = others + [b]
                        new_score = calc_score(new_chosen)
                        if new_score > best_score + 1e-12:
                            best = new_chosen
                            best_score = new_score
                            improved = True
                            break
                if improved:
                    break
    
    return best
