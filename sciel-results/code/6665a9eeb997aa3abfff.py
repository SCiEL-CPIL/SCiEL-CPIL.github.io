import time
import random
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    primes = _sieve(1000)
    all_primes = primes
    N = 100000
    
    best_L = 0
    best_classes = []
    
    # Pre-compute costs
    costs = {p: Fraction(1, p) for p in all_primes}
    
    def compute_L(covered):
        L = 0
        for i in range(1, N + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def run_greedy(window_size, prefer_small_on_tie, use_efficiency, deadline, seed=0):
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = bytearray(N + 1)
        used_primes = set()
        budget = Fraction(1)
        classes = []
        
        # Optimization: Keep track of the next uncovered number to skip over covered blocks
        # But since we need to check coverage for future candidates, a simple loop with 'continue' is fine
        # unless the blocks are very long. Given the density, let's stick to the loop but optimize the inner check.
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            # n is uncovered
            best_p = None
            best_r = 0
            best_score = -1
            best_cnt = 0
            best_cost = Fraction(0)
            
            # Candidates: unused primes that fit in budget
            for p in all_primes:
                if p in used_primes:
                    continue
                cost = costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                # Count new coverage in [n, n+window_size]
                cnt = 0
                if p > window_size:
                    if not covered[n]:
                        cnt = 1
                else:
                    max_k = n + window_size
                    if max_k > N:
                        max_k = N
                    # Optimization: Use a while loop or list comprehension?
                    # List comprehension might be faster in PyPy/CPython for small ranges?
                    # Let's stick to the loop but ensure it's tight.
                    for k in range(n, max_k + 1, p):
                        if not covered[k]:
                            cnt += 1
                
                # Calculate Score
                if use_efficiency:
                    score = cnt * p
                else:
                    score = cnt
                    if prefer_small_on_tie:
                        score += (1000 - p)
                    else:
                        score += p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
                    best_cnt = cnt
                    best_cost = cost
            
            if best_p is None:
                # Can't cover n
                break
            
            # Commit to best_p
            used_primes.add(best_p)
            budget -= best_cost
            classes.append([best_p, best_r])
            
            # Mark covered
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, N + 1, best_p):
                covered[k] = 1
        
        L = compute_L(covered)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        
        return L

    def run_greedy_v2(deadline, seed=0):
        """
        Variant that uses a different scoring heuristic:
        Prioritize primes that cover the current gap AND have high "future potential"
        (i.e., small p, which covers more dense numbers, but costs more).
        Actually, let's try a "lookahead" approach where we evaluate the gap after covering n.
        """
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = bytearray(N + 1)
        used_primes = set()
        budget = Fraction(1)
        classes = []
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in all_primes:
                if p in used_primes:
                    continue
                cost = costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                
                # Heuristic: Prefer primes that result in a longer run of covered numbers starting from n.
                # Let's simulate the coverage up to n + 1000 and count the length of the contiguous block starting at n.
                
                # This is expensive if done naively.
                # Let's just count total new coverage in [n, n+2000] but weight later coverage less?
                # Or just use the previous efficient score but with a different window.
                
                # Let's try: score = cnt * (1 + 1/p) ? No.
                
                # Let's try a different metric:
                # How many "uncovered" numbers in [n, n+p] does this cover?
                # This is just cnt in [n, n+p].
                
                cnt = 0
                max_k = min(n + p, N)
                for k in range(n, max_k + 1, p):
                    if not covered[k]:
                        cnt += 1
                
                # If cnt is same, prefer larger p (cheaper)?
                # Or smaller p (denser)?
                # Smaller p covers n, n+p, n+2p...
                # Larger p covers n, n+p... (sparser)
                # But we only care about contiguous coverage from 1.
                # If we pick p, we cover n. The next uncovered might be n+1.
                # If p is small, n+p is close, so it might cover n+1 if (n+1)%p == r? No, r is fixed.
                # r = n % p.
                # n+1 is covered if (n+1) % p == r => n % p + 1 == n % p (mod p) => 1 == 0 (mod p). Impossible for p>1.
                # So picking a prime for n NEVER covers n+1 directly unless n+1 was already covered by another prime.
                # Wait.
                # If we pick p with r = n % p, then n is covered.
                # n+1 is covered if there exists a previously picked prime q with (n+1) % q == r_q.
                # So the choice of p for n does NOT affect coverage of n+1, n+2, etc., EXCEPT through the "used_primes" constraint (none, since p is new) and budget (affects future choices).
                # AND it covers future numbers n+p, n+2p, etc.
                
                # So the "cnt" in [n, n+window] is the only forward-looking benefit.
                # The immediate benefit is just covering n (which is 1).
                # So we want to maximize future coverage per cost.
                # score = cnt / cost = cnt * p.
                # This is what use_efficiency=True does.
                
                # Let's try a variation:
                # Score = cnt * p - lambda * p (penalize large p? No, large p is cheap).
                # Score = cnt * p.
                
                # Let's try to break ties by preferring primes that leave more "flexibility" in budget?
                # i.e., prefer smaller cost (larger p).
                
                score = cnt * p
                # Tie breaker: prefer larger p (smaller cost) to save budget?
                # Or smaller p to cover more dense future points?
                # If cnt is 1 (only n is covered in window), then score = p.
                # So we prefer largest p.
                # This saves budget.
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget -= costs[best_p]
            classes.append([best_p, best_r])
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, N + 1, best_p):
                covered[k] = 1
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        return L

    def run_greedy_v3(deadline, seed=0):
        """
        Variant that uses a "density" score: cnt / (1 + log(p)) or similar.
        Also tries to maximize the minimum gap between uncovered numbers in the window.
        """
        nonlocal best_L, best_classes
        
        rng = random.Random(seed)
        covered = bytearray(N + 1)
        used_primes = set()
        budget = Fraction(1)
        classes = []
        
        for n in range(1, N + 1):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in all_primes:
                if p in used_primes:
                    continue
                cost = costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                
                # Count new coverage in [n, n+5000]
                cnt = 0
                max_k = min(n + 5000, N)
                for k in range(n, max_k + 1, p):
                    if not covered[k]:
                        cnt += 1
                
                # Score: cnt * p (efficiency) but with a penalty for very large p if cnt is low?
                # If cnt == 1, score = p. We want to save budget, so large p is good.
                # If cnt > 1, large p might not cover as many.
                
                # Let's try: score = cnt * (p ** 0.5)
                # This balances density (small p) and cost (large p).
                
                score = cnt * (p ** 0.5)
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget -= costs[best_p]
            classes.append([best_p, best_r])
            
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, N + 1, best_p):
                covered[k] = 1
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        return L

    # Run a mix of strategies
    strategies = [
        # (window, prefer_small, use_eff, seed)
        (1000, True, True, 1),
        (1000, False, True, 2),
        (5000, True, True, 3),
        (5000, False, True, 4),
        (10000, True, True, 5),
        (10000, False, True, 6),
        (2000, True, True, 7),
        (2000, False, True, 8),
        (500, True, True, 9),
        (500, False, True, 10),
        # Add some non-efficiency strategies
        (1000, True, False, 11),
        (1000, False, False, 12),
        (5000, True, False, 13),
        (5000, False, False, 14),
    ]
    
    # First pass: fixed strategies
    for w, ps, ue, s in strategies:
        if time.time() > deadline - 10:
            break
        run_greedy(w, ps, ue, deadline, s)
    
    # Second pass: v2 (which is effectively w=p, eff=True, tie_breaker=larger_p)
    run_greedy_v2(deadline)
    
    # Third pass: v3 (density score)
    run_greedy_v3(deadline)
    
    # Fourth pass: Randomized perturbations
    if time.time() < deadline - 10:
        for i in range(30):
            if time.time() > deadline - 5:
                break
            w = random.choice([500, 1000, 2000, 5000, 10000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            s = 100 + i
            run_greedy(w, ps, ue, deadline, s)
            
    # Fifth pass: Try to improve the best solution found so far?
    # This is complex. Instead, let's just run more variants.
    
    if not best_classes:
        # Fallback
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
        
    return best_classes
