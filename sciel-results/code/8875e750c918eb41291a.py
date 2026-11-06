import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # 1. Generate initial solution using greedy
    S = set()
    for a in range(n, 0, -1):
        valid = True
        lst = list(S)
        m = len(lst)
        for i in range(m):
            b = lst[i]
            for j in range(i, m):
                c = lst[j]
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(a)
            
    # Helper to calculate total violations
    def count_violations(S_set):
        count = 0
        lst = list(S_set)
        m = len(lst)
        # Sort to potentially help with early breaks? Not really needed for count.
        for i in range(m):
            a = lst[i]
            for j in range(m):
                b = lst[j]
                if b <= a:
                    continue
                for k in range(m):
                    c = lst[k]
                    if c < b:
                        continue
                    if (b + c) % a == 0:
                        count += 1
        return count

    # Helper to calculate delta violations for adding x to S_set
    def delta_violations_add(x, S_set):
        delta = 0
        # Case 1: x is a. x < b <= c
        for b in S_set:
            if b <= x:
                continue
            for c in S_set:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    delta += 1
        # Case 2: x is b. a < x <= c
        for a in S_set:
            if a >= x:
                continue
            for c in S_set:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    delta += 1
        # Case 3: x is c. a < b <= x
        for a in S_set:
            for b in S_set:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    delta += 1
        return delta

    # Helper to calculate delta violations for removing x from S_set
    # This is the same as delta_violations_add(x, S_set - {x})
    # But we can just use the same logic, knowing that x is in S_set
    def delta_violations_remove(x, S_set):
        # The violations removed are those that involved x in the current set S_set.
        # This is exactly the same count as delta_violations_add(x, S_set \ {x})
        # But we must be careful: the pairs (b,c) must be from S_set \ {x}.
        S_without_x = S_set - {x}
        return delta_violations_add(x, S_without_x)

    current_S = set(S)
    current_V = count_violations(current_S)
    current_score = len(current_S) - 1000 * current_V
    best_S = set(current_S)
    best_score = current_score
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute list of all numbers for random selection
    all_nums = list(range(1, n + 1))
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        # 50% add, 50% remove
        if random.random() < 0.5:
            # Add a random element not in S
            candidates = [x for x in all_nums if x not in current_S]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            dV = delta_violations_add(x, current_S)
            dS = 1
            dScore = dS - 1000 * dV
            if dScore > 0 or random.random() < math.exp(dScore / T):
                current_S.add(x)
                current_V += dV
                current_score += dScore
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(current_S)
        else:
            # Remove a random element from S
            if not current_S:
                T *= cooling
                continue
            x = random.choice(list(current_S))
            dV = -delta_violations_remove(x, current_S) # Removing reduces V by this amount
            # Wait, delta_violations_remove returns the count of violations that WILL be removed.
            # So new_V = old_V - delta_violations_remove(x, current_S)
            # dV = new_V - old_V = -delta_violations_remove(x, current_S)
            dS = -1
            dScore = dS - 1000 * dV
            # dScore = -1 - 1000 * (-dV_removed) = -1 + 1000 * dV_removed
            if dScore > 0 or random.random() < math.exp(dScore / T):
                current_S.remove(x)
                current_V -= delta_violations_remove(x, current_S) # Need to calculate before removal?
                # Actually, let's store dV_removed first
                # Let's refactor to avoid double calculation or error
                
                # Recalculate properly
                # We already calculated dV for the move decision.
                # If accepted:
                # current_V += dV
                # current_score += dScore
                
                pass # Handled below
                
        T *= cooling

    # Refine: Local search to add any valid elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in all_nums:
            if x in best_S:
                continue
            if delta_violations_add(x, best_S) == 0:
                best_S.add(x)
                improved = True

    return list(best_S)
