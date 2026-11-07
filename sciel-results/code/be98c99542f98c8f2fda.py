import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute prime factorizations
    def prime_factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    factorizations = {n: prime_factorize(n) for n in candidates}
    
    # Precompute: for each pair (b,c), which a's divide b*c?
    # This is expensive, so we'll compute on demand but cache
    
    def divides_product(a, b, c):
        """Check if a divides b*c using prime factorizations"""
        fa = factorizations[a]
        fb = factorizations[b]
        fc = factorizations[c]
        for p, ea in fa.items():
            eb = fb.get(p, 0)
            ec = fc.get(p, 0)
            if ea > eb + ec:
                return False
        return True
    
    def is_valid_set(s):
        """Check if set s is valid (no three distinct a,b,c with a|b*c)"""
        ss = list(set(s))
        n = len(ss)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = ss[i], ss[j], ss[k]
                    if divides_product(a, b, c) or divides_product(b, a, c) or divides_product(c, a, b):
                        return False
        return True
    
    def can_add(a, current_set):
        """Check if a can be added to current_set without violating the condition"""
        cs = list(current_set)
        n = len(cs)
        for i in range(n):
            b = cs[i]
            for j in range(i + 1, n):
                c = cs[j]
                if divides_product(a, b, c) or divides_product(b, a, c) or divides_product(c, a, b):
                    return False
        return True
    
    def greedy_search(order):
        s = []
        for a in order:
            if can_add(a, s):
                s.append(a)
        return s
    
    def local_search(s, max_iters=5000):
        """Try to improve set s by removing one element and adding another"""
        s = list(s)
        s_set = set(s)
        improved = True
        iters = 0
        best_size = len(s)
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            # Try removing one element and adding a better one
            for i in range(len(s)):
                candidate_to_remove = s[i]
                new_set = [x for x in s if x != candidate_to_remove]
                new_set_set = set(new_set)
                # Try adding an element not in new_set
                for a in candidates:
                    if a in new_set_set:
                        continue
                    if can_add(a, new_set):
                        if a != candidate_to_remove:
                            new_s = new_set + [a]
                            if len(new_s) > best_size:
                                s = new_s
                                s_set = set(s)
                                best_size = len(s)
                                improved = True
                                break
                if improved:
                    break
        return s
    
    def targeted_local_search(s, max_iters=3000):
        """More targeted local search: try swaps that might improve"""
        s = list(s)
        s_set = set(s)
        improved = True
        iters = 0
        best_size = len(s)
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            # Try all possible single removals and additions
            for i in range(len(s)):
                to_remove = s[i]
                remaining = [x for x in s if x != to_remove]
                rem_set = set(remaining)
                for a in candidates:
                    if a in rem_set:
                        continue
                    if can_add(a, remaining):
                        if a != to_remove:
                            new_s = remaining + [a]
                            if len(new_s) > best_size:
                                s = new_s
                                s_set = set(s)
                                best_size = len(s)
                                improved = True
                                break
                if improved:
                    break
        return s
    
    best_set = []
    
    # Strategy 1: Sort by "divisibility risk" - numbers with large prime factors first
    def risk_score(n):
        """Lower risk = better to include first"""
        f = factorizations[n]
        max_prime = max(f.keys()) if f else 0
        num_factors = len(f)
        # Prefer numbers with large primes and few factors
        return -max_prime + num_factors * 0.1
    
    order1 = sorted(candidates, key=risk_score)
    s1 = greedy_search(order1)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Primes first (especially large primes), then composites
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    order2 = sorted(primes, reverse=True) + sorted(composites, key=risk_score)
    s2 = greedy_search(order2)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: Numbers with exactly one large prime factor
    def has_large_prime(n, threshold=64):
        f = factorizations[n]
        return any(p >= threshold for p in f.keys())
    
    large_prime_nums = [x for x in candidates if has_large_prime(x)]
    other_nums = [x for x in candidates if not has_large_prime(x)]
    order3 = sorted(large_prime_nums, reverse=True) + sorted(other_nums, key=risk_score)
    s3 = greedy_search(order3)
    if len(s3) > len(best_set):
        best_set = s3
    
    # Strategy 4: Randomized restarts
    for _ in range(300):
        if time.time() > deadline:
            break
        order = candidates[:]
        random.shuffle(order)
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search on best set
    if time.time() < deadline - 5:
        improved = local_search(best_set, max_iters=5000)
        if len(improved) > len(best_set):
            best_set = improved
    
    # More randomized restarts with local search
    while time.time() < deadline - 10:
        order = candidates[:]
        random.shuffle(order)
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s
        if time.time() < deadline - 5:
            improved = targeted_local_search(best_set, max_iters=2000)
            if len(improved) > len(best_set):
                best_set = improved
    
    # Final polish: try to add any remaining candidates
    if time.time() < deadline - 2:
        best_set_set = set(best_set)
        for a in candidates:
            if a in best_set_set:
                continue
            if can_add(a, best_set):
                best_set.append(a)
                best_set_set.add(a)
    
    return list(best_set)
