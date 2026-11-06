"""Improved program: faster validity checks, smarter ordering, and more thorough search."""
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def can_add(x, S):
        """Check if x can be added to set S without violating constraints.
        
        Two conditions must hold:
        1. x does not divide the sum of any 3 distinct elements in S.
        2. No element a in S divides the sum of x plus any 2 distinct elements in S\{a}.
        """
        n = len(S)
        
        # Condition 1: x must not divide sum of any 3 distinct elements in S
        if n >= 3:
            # Precompute all triple sums? No, just check divisibility.
            # Optimization: if x > max possible triple sum, skip. Max triple sum = 3*192 = 576.
            if x <= 576:
                for i in range(n):
                    si = S[i]
                    for j in range(i+1, n):
                        sj = S[j]
                        for k in range(j+1, n):
                            if (si + sj + S[k]) % x == 0:
                                return False
        
        # Condition 2: For each a in S, a must not divide (x + b + c) for any distinct b,c in S\{a}
        for a in S:
            # If a > x + 2*192 = x+384, then x+b+c < a, so (x+b+c)%a != 0 (since x+b+c > 0)
            if a > x + 384:
                continue
            # Find b, c in S such that b != a, c != a, b != c, and (x+b+c) % a == 0
            # We need to check pairs (b,c) from S\{a}
            others = [v for v in S if v != a]
            m = len(others)
            if m < 2:
                continue
            for i in range(m):
                bi = others[i]
                for j in range(i+1, m):
                    cj = others[j]
                    if (x + bi + cj) % a == 0:
                        return False
        return True

    # Greedy construction with smart ordering
    # Start with large numbers (they're less likely to divide small sums)
    # Then try to add smaller numbers that fit
    S = []
    
    # Phase 1: Greedy add from large to small
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.append(x)
    
    # Phase 2: Try to add remaining numbers in increasing order (smaller numbers might fit now)
    added = True
    while added and time.time() < deadline:
        added = False
        S_set = set(S)
        for v in range(1, 193):
            if time.time() > deadline:
                break
            if v not in S_set:
                if can_add(v, S):
                    S.append(v)
                    S_set.add(v)
                    added = True
    
    # Phase 3: Local search - try swaps to potentially increase size
    # Try removing one element and adding two new ones, or just swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_set = set(S)
        
        # Try to add any missing number
        for v in range(1, 193):
            if time.time() > deadline:
                break
            if v not in S_set:
                if can_add(v, S):
                    S.append(v)
                    S_set.add(v)
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swaps: remove u, add v
        # Only do this if it might help (hard to know, so try a limited number)
        swap_attempts = 0
        while swap_attempts < 200 and time.time() < deadline:
            swap_attempts += 1
            if not S:
                break
            u_idx = (swap_attempts * 7) % len(S)  # Deterministic pseudo-random
            u = S[u_idx]
            
            # Try to find a v not in S that can replace u
            found = False
            for v in range(1, 193):
                if time.time() > deadline:
                    break
                if v in S_set:
                    continue
                S_temp = S[:u_idx] + S[u_idx+1:]
                if can_add(v, S_temp):
                    S = S_temp
                    S.append(v)
                    S_set.remove(u)
                    S_set.add(v)
                    improved = True
                    found = True
                    break
            
            if found:
                break
    
    return S
