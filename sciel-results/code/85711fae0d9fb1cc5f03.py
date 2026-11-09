import time
import random

def construct():
    N = 256
    deadline = time.time() + 55.0  # Safety margin

    # Precompute all 3-term arithmetic progressions (a, b, c) with a < b < c
    aps = []
    for b in range(2, N + 1):
        for a in range(1, b):
            c = 2 * b - a
            if c <= N:
                aps.append((a, b, c))

    # For each element x, store which APs contain it and its role (0=a, 1=b, 2=c)
    ap_by_elem = {i: [] for i in range(1, N + 1)}
    for a, b, c in aps:
        ap_by_elem[a].append((a, b, c, 0))
        ap_by_elem[b].append((a, b, c, 1))
        ap_by_elem[c].append((a, b, c, 2))

    def can_add_fast(S, x):
        """Check if x can be added to S without creating a 3-AP."""
        for a, b, c, role in ap_by_elem[x]:
            if role == 0:  # x is the first term
                if b in S and c in S:
                    return False
            elif role == 1:  # x is the middle term
                if a in S and c in S:
                    return False
            else:  # x is the last term
                if a in S and b in S:
                    return False
        return True

    def score(S):
        return sum(1.0 / x for x in S)

    # Base-11 digit-free construction: select numbers with no zero digits in base 11
    def has_no_zero_digits_base11(n):
        while n > 0:
            if n % 11 == 0:
                return False
            n //= 11
        return n == 0  # n was 0 initially

    # Initial greedy construction using base-11 hint
    S = set()
    # First, add all numbers with no zero digits in base 11 (sorted by reciprocal weight)
    base11_candidates = [x for x in range(1, N + 1) if has_no_zero_digits_base11(x)]
    base11_candidates.sort(key=lambda x: 1.0 / x, reverse=True)  # Largest reciprocal first
    
    for x in base11_candidates:
        if can_add_fast(S, x):
            S.add(x)
    
    # Then greedily add remaining numbers by reciprocal weight
    remaining = [x for x in range(1, N + 1) if x not in S]
    remaining.sort(key=lambda x: 1.0 / x, reverse=True)
    for x in remaining:
        if can_add_fast(S, x):
            S.add(x)

    best_S = set(S)
    best_score = score(S)
    current_score = best_score

    # Local search: try removing one element and refilling greedily
    while time.time() < deadline:
        improved = False
        
        # Try to improve by removing one element and refilling
        elements_to_remove = list(S)
        random.shuffle(elements_to_remove)
        
        for x in elements_to_remove:
            if time.time() > deadline:
                break
            
            S.remove(x)
            
            # Refill greedily by reciprocal weight
            refill_changed = True
            while refill_changed and time.time() < deadline:
                refill_changed = False
                # Find best candidate to add
                best_candidate = None
                best_weight = 0.0
                for y in range(1, N + 1):
                    if y not in S:
                        w = 1.0 / y
                        if w > best_weight and can_add_fast(S, y):
                            best_weight = w
                            best_candidate = y
                if best_candidate is not None:
                    S.add(best_candidate)
                    refill_changed = True
            
            new_score = score(S)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_S = set(S)
                improved = True
                break
            else:
                # Restore x
                S.add(x)
                # Refill to restore maximal state
                refill_changed = True
                while refill_changed and time.time() < deadline:
                    refill_changed = False
                    best_candidate = None
                    best_weight = 0.0
                    for y in range(1, N + 1):
                        if y not in S:
                            w = 1.0 / y
                            if w > best_weight and can_add_fast(S, y):
                                best_weight = w
                                best_candidate = y
                    if best_candidate is not None:
                        S.add(best_candidate)
                        refill_changed = True
        
        if not improved:
            # Try 2-removal kick
            if len(S) > 2:
                sorted_S = sorted(S, key=lambda x: 1.0 / x)
                half = len(sorted_S) // 2
                candidates = sorted_S[half:]
                if len(candidates) >= 2:
                    r1, r2 = random.sample(candidates, 2)
                    S.remove(r1)
                    S.remove(r2)
                    
                    refill_changed = True
                    while refill_changed and time.time() < deadline:
                        refill_changed = False
                        best_candidate = None
                        best_weight = 0.0
                        for y in range(1, N + 1):
                            if y not in S:
                                w = 1.0 / y
                                if w > best_weight and can_add_fast(S, y):
                                    best_weight = w
                                    best_candidate = y
                        if best_candidate is not None:
                            S.add(best_candidate)
                            refill_changed = True
                    
                    new_score = score(S)
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_S = set(S)
                    else:
                        # Restore
                        S.add(r1)
                        S.add(r2)
                        refill_changed = True
                        while refill_changed and time.time() < deadline:
                            refill_changed = False
                            best_candidate = None
                            best_weight = 0.0
                            for y in range(1, N + 1):
                                if y not in S:
                                    w = 1.0 / y
                                    if w > best_weight and can_add_fast(S, y):
                                        best_weight = w
                                        best_candidate = y
                            if best_candidate is not None:
                                S.add(best_candidate)
                                refill_changed = True

    return list(best_S)
