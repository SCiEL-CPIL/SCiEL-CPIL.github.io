"""Rewritten program: DFS with 2-term number-theoretic completion."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    
    # Greedy initial solution
    r = target
    greedy = []
    while r and len(greedy) < 256:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy.append(d)
        r -= Fraction(1, d)
    
    best = greedy if r == 0 else None
    best_max = max(greedy) if best else 10000000
    
    def complete_2_terms(p, q, min_den, max_allowed, current_set):
        """Find b,c with 1/b+1/c=p/q, b>=min_den, c>b, b,c<=max_allowed, not in current_set."""
        if p == 0:
            return []
        q2 = q * q
        if q2 > 10**10:
            return []
        results = []
        for i in range(1, isqrt(q2) + 1):
            if q2 % i == 0:
                for u, v in [(i, q2//i), (q2//i, i)]:
                    if u > v:
                        continue
                    nb, nc = u + q, v + q
                    if nb % p != 0 or nc % p != 0:
                        continue
                    b, c = nb // p, nc // p
                    if b < min_den or c <= b or b > max_allowed or c > max_allowed:
                        continue
                    if b in current_set or c in current_set:
                        continue
                    results.append((b, c))
        return results
    
    def dfs(remaining, min_den, current, max_allowed):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        if remaining == 0:
            if current:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best = list(current)
            return
        if len(current) >= 256:
            return
        
        p, q = remaining.numerator, remaining.denominator
        
        # If 2 terms left, use number-theoretic completion
        if len(current) == 254:
            pairs = complete_2_terms(p, q, min_den, max_allowed, set(current))
            for b, c in pairs:
                cand = current + [b, c]
                m = max(cand)
                if m < best_max:
                    best_max = m
                    best = list(cand)
            return
        
        # If 1 term left
        if len(current) == 255:
            if p == 1:
                d = q
                if d >= min_den and d <= max_allowed and d not in set(current):
                    cand = current + [d]
                    m = max(cand)
                    if m < best_max:
                        best_max = m
                        best = list(cand)
            return
        
        d_min = max(min_den, (q + p - 1) // p)
        max_terms = 256 - len(current)
        if remaining > Fraction(max_terms, d_min):
            return
        
        d_upper = min(max_allowed, 10000000)
        if d_min > d_upper:
            return
        
        d = d_min
        count = 0
        while d <= d_upper and count < 50000:
            if time.time() > deadline:
                return
            count += 1
            term = Fraction(1, d)
            if term > remaining:
                d += 1
                continue
            nr = remaining - term
            if nr < 0:
                d += 1
                continue
            current.append(d)
            dfs(nr, d + 1, current, max_allowed)
            current.pop()
            d += 1
    
    dfs(target, 1, [], best_max)
    
    if best is None:
        return greedy
    return best
# EVOLVE-BLOCK-END
