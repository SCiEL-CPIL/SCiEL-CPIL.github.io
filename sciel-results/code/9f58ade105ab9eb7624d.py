def construct():
    import time
    from math import gcd
    
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s
    
    # Precompute prime factors for each number 1..511
    def prime_factors(n):
        if n == 1:
            return frozenset()
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return frozenset(factors)
    
    # Precompute weights and prime factors
    weights = {}
    pfactors = {}
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        pfactors[a] = prime_factors(a)
    
    # Greedy: sort by weight descending, pick if coprime with all chosen
    def greedy_select():
        chosen = []
        used_primes = set()
        for a in sorted(range(1, 512), key=lambda x: weights[x], reverse=True):
            if a == 1:
                chosen.append(a)
                continue
            if pfactors[a] & used_primes:
                continue
            chosen.append(a)
            used_primes |= pfactors[a]
        return chosen
    
    best_set = greedy_select()
    best_score = sum(weights[a] for a in best_set)
    
    # Local search: try to swap out a low-weight element for a high-weight one
    def try_improve(chosen, max_iters=50000):
        chosen_set = set(chosen)
        used_primes = set()
        for a in chosen:
            used_primes |= pfactors[a]
        improved = True
        iters = 0
        while improved and iters < max_iters and (time.time() - start_time) < DEADLINE:
            improved = False
            iters += 1
            # Try removing each element and see if we can add a better one
            for i, a in enumerate(list(chosen)):
                if a == 1:
                    continue
                # Remove a
                new_used = set()
                for b in chosen:
                    if b != a:
                        new_used |= pfactors[b]
                # Try adding each candidate not in chosen
                best_candidate = None
                best_gain = 0
                for c in range(511, 0, -1):
                    if c in chosen_set and c != a:
                        continue
                    if c == 1 and 1 in chosen_set:
                        continue
                    if c == a:
                        continue
                    if c == 1:
                        if 1 not in chosen_set:
                            gain = weights[1] - weights[a]
                            if gain > best_gain:
                                best_gain = gain
                                best_candidate = c
                        continue
                    if pfactors[c] & new_used:
                        continue
                    gain = weights[c] - weights[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c
                if best_candidate is not None:
                    chosen.remove(a)
                    chosen.append(best_candidate)
                    chosen_set = set(chosen)
                    improved = True
                    # Recompute used_primes
                    used_primes = set()
                    for b in chosen:
                        used_primes |= pfactors[b]
                    return True
        return improved
    
    # Try local search
    improved = try_improve(best_set)
    if improved:
        best_score = sum(weights[a] for a in best_set)
    
    # Try multiple greedy restarts with different orderings
    for _ in range(20):
        if (time.time() - start_time) >= DEADLINE:
            break
        # Random perturbation: start with a random subset
        import random
        random.seed(_ * 12345)
        chosen = [1]  # always include 1
        used_primes = set()
        # Add some random high-weight elements
        candidates = list(range(511, 1, -1))
        random.shuffle(candidates)
        for c in candidates[:10]:
            if pfactors[c] & used_primes:
                continue
            chosen.append(c)
            used_primes |= pfactors[c]
        # Then complete greedily
        for c in sorted(range(511, 0, -1), key=lambda x: weights[x], reverse=True):
            if c in chosen:
                continue
            if c == 1:
                continue
            if pfactors[c] & used_primes:
                continue
            chosen.append(c)
            used_primes |= pfactors[c]
        score = sum(weights[a] for a in chosen)
        if score > best_score:
            best_score = score
            best_set = list(chosen)
        # Try local search on this
        if (time.time() - start_time) < DEADLINE:
            if try_improve(best_set, max_iters=10000):
                best_score = sum(weights[a] for a in best_set)
    
    return best_set
