import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    
    def check_add_any(S, x):
        """Check if x can be added to S without creating a forbidden triple."""
        # x is smallest: 1/x = 1/b + 1/c => c = x*b/(b-x)
        for b in S:
            if b <= x:
                continue
            den = b - x
            if den == 0:
                continue
            num = x * b
            if num % den != 0:
                continue
            c = num // den
            if c in S and c != b and c != x:
                return False
        # x is largest: 1/a = 1/x + 1/c => c = a*x/(x-a)
        for a in S:
            if a >= x:
                continue
            den = x - a
            if den == 0:
                continue
            num = a * x
            if num % den != 0:
                continue
            c = num // den
            if c in S and c != a and c != x:
                return False
        return True

    def greedy_backward():
        S = set()
        for x in range(180, 0, -1):
            if check_add_any(S, x):
                S.add(x)
        return S

    def greedy_forward():
        S = set()
        for x in range(1, 181):
            if check_add_any(S, x):
                S.add(x)
        return S

    def greedy_random():
        items = list(range(1, 181))
        random.shuffle(items)
        S = set()
        for x in items:
            if check_add_any(S, x):
                S.add(x)
        return S

    def greedy_weighted(weights):
        """Greedy with weighted scoring."""
        items = list(range(1, 181))
        items.sort(key=lambda x: -weights[x])
        S = set()
        for x in items:
            if check_add_any(S, x):
                S.add(x)
        return S

    # Start with greedy solutions
    best = greedy_backward()
    best_size = len(best)
    
    S2 = greedy_forward()
    if len(S2) > best_size:
        best = S2
        best_size = len(S2)

    # Random restarts
    count = 0
    while time.time() - start_time < deadline - 15:
        S_r = greedy_random()
        if len(S_r) > best_size:
            best = S_r
            best_size = len(S_r)
        count += 1
        if count > 500:
            break

    # Local search: try swapping elements
    if time.time() - start_time < deadline - 5:
        improved = True
        while improved and time.time() - start_time < deadline - 3:
            improved = False
            # Try removing each element and adding a different one
            for x in list(best):
                if time.time() - start_time > deadline - 3:
                    break
                best.remove(x)
                # Find best replacement
                best_y = None
                best_score = 0
                for y in range(1, 181):
                    if y in best or y == x:
                        continue
                    if check_add_any(best, y):
                        # Score based on how many other elements it conflicts with
                        conflicts = 0
                        for z in best:
                            if z != y:
                                # Check if adding z would conflict with y
                                pass
                        # Simple: just take the first valid one
                        best_y = y
                        break
                if best_y is not None:
                    best.add(best_y)
                    if len(best) > best_size:
                        best_size = len(best)
                        improved = True
                else:
                    best.add(x)  # Revert

    return list(best)
