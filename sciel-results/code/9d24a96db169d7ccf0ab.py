"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    n = 128
    
    # Precompute: for each pair (a,b) with a<b, store a+b
    # A set S is valid iff: for all distinct a,b,c in S, NOT (a+b in S AND a+c in S AND b+c in S)
    # Equivalently: for all pairs (a,b) in S with a+b in S, no c in S\{a,b} has both a+c in S and b+c in S.
    
    def is_valid_set(S):
        """Check validity of a set S."""
        S_arr = np.array(sorted(S), dtype=np.int32)
        k = len(S_arr)
        if k < 3:
            return True
        S_set = set(S)
        for i in range(k):
            a = S_arr[i]
            for j in range(i + 1, k):
                b = S_arr[j]
                ab = a + b
                if ab not in S_set:
                    continue
                # Need: no c in S, c != a, c != b, with a+c in S and b+c in S
                for kk in range(k):
                    c = S_arr[kk]
                    if c == a or c == b:
                        continue
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True
    
    def count_violations(S):
        """Count number of forbidden triples in S."""
        S_set = set(S)
        S_arr = sorted(S)
        k = len(S_arr)
        count = 0
        for i in range(k):
            a = S_arr[i]
            for j in range(i + 1, k):
                b = S_arr[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for kk in range(k):
                    c = S_arr[kk]
                    if c == a or c == b:
                        continue
                    if (a + c) in S_set and (b + c) in S_set:
                        count += 1
        return count
    
    def can_add_fast(x, S_set, S_arr, k):
        """Check if x can be added to S without creating a forbidden triple.
        New forbidden triples must include x. So for distinct a,b in S:
        check if (x+a in S) and (x+b in S) and (a+b in S)."""
        for i in range(k):
            a = S_arr[i]
            if a == x:
                continue
            xa = x + a
            if xa not in S_set:
                continue
            for j in range(i + 1, k):
                b = S_arr[j]
                if b == x:
                    continue
                if (x + b) in S_set and (a + b) in S_set:
                    return False
        return True
    
    # Strategy: Multi-start with different initial seeds, then local search
    # Key structural insight: use density-based pruning
    
    best_result = None
    best_size = 0
    
    # Multiple random restarts with different strategies
    num_restarts = 8
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Upper half
            S = set(range(65, n + 1))
        elif restart == 1:
            # Upper half minus some, plus some lower
            S = set(range(65, n + 1))
            for _ in range(5):
                S.discard(rng.randint(65, 128))
        elif restart == 2:
            # Random subset of size ~80
            S = set(rng.sample(range(1, n + 1), 80))
        elif restart == 3:
            # Even numbers only in upper + some lower
            S = set(range(66, n + 1, 2))
        elif restart == 4:
            # Multiples of 3 in upper + some
            S = set(range(66, n + 1, 3))
        elif restart == 5:
            # Top 40 + random from middle
            S = set(range(89, n + 1))
            S.update(rng.sample(range(33, 89), 20))
        elif restart == 6:
            # Fibonacci-like spacing
            S = set()
            x = 1
            while x <= n:
                S.add(x)
                x += max(2, x // 3)
        else:
            # Random
            S = set(rng.sample(range(1, n + 1), rng.randint(60, 90)))
        
        # Remove violations from initial set
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            S_set = set(S)
            S_arr = sorted(S)
            k = len(S_arr)
            # Find a violation and remove one element
            for i in range(k):
                a = S_arr[i]
                for j in range(i + 1, k):
                    b = S_arr[j]
                    ab = a + b
                    if ab not in S_set:
                        continue
                    for kk in range(k):
                        c = S_arr[kk]
                        if c == a or c == b:
                            continue
                        if (a + c) in S_set and (b + c) in S_set:
                            # Remove the element that causes most conflicts
                            # Simple: remove c (or a, or b - try all three)
                            for rem in [a, b, c]:
                                S_try = S_set - {rem}
                                if count_violations(S_try) < count_violations(S_set):
                                    S = S_try
                                    changed = True
                                    break
                    if changed:
                        break
                if changed:
                    break
        
        # Local search: try to add elements
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            S_arr = sorted(S_set)
            k = len(S_arr)
            for x in range(1, n + 1):
                if x in S_set:
                    continue
                if can_add_fast(x, S_set, S_arr, k):
                    S_set.add(x)
                    improved = True
            # Try swaps
            if not improved and time.time() < deadline - 2:
                not_in_S = [x for x in range(1, n + 1) if x not in S_set]
                for y in list(S_set):
                    if time.time() > deadline - 2:
                        break
                    for x in not_in_S:
                        if time.time() > deadline - 2:
                            break
                        new_S = (S_set - {y}) | {x}
                        if count_violations(new_S) == 0:
                            S_set = new_S
                            not_in_S.remove(x)
                            not_in_S.append(y)
                            improved = True
                            break
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_result = set(S_set)
    
    # Final validation and cleanup
    if best_result is None:
        best_result = set(range(65, n + 1))
    
    # Verify
    if not is_valid_set(best_result):
        # Fallback
        best_result = set(range(65, n + 1))
    
    result = sorted(best_result)
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
