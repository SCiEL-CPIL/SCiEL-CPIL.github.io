import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    candidates = list(range(1, N + 1))
    
    def is_valid_set(s):
        """Check if set s is pairwise coprime."""
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def greedy_construct(order):
        """Greedy construction based on given order."""
        s = set()
        for x in order:
            valid = True
            for y in s:
                if math.gcd(x, y) != 1:
                    valid = False
                    break
            if valid:
                s.add(x)
        return s

    def local_search(s):
        """
        Perform local search to improve the set.
        Moves:
        1. Add: Add x if valid.
        2. Swap 1-for-1: Replace y with x if sum increases and valid.
        3. Swap 2-for-1: Replace {a, b} with c if sum increases and valid.
        4. Swap 1-for-2: Replace a with {b, c} if sum increases and valid.
        """
        improved = True
        # Limit iterations to prevent infinite loops in case of oscillation
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            current_sum = sum(s)
            
            # 1. Try adding elements
            for x in range(N, 0, -1):
                if x in s:
                    continue
                valid = True
                for y in s:
                    if math.gcd(x, y) != 1:
                        valid = False
                        break
                if valid:
                    s.add(x)
                    improved = True
                    # Restart loop to re-evaluate with new element
                    break 
            
            if improved:
                continue

            # 2. Try 1-for-1 swaps
            for y in list(s):
                for x in range(N, 0, -1):
                    if x == y or x in s:
                        continue
                    if x <= y:
                        continue # Only improve sum
                    new_s = s - {y}
                    valid = True
                    for z in new_s:
                        if math.gcd(x, z) != 1:
                            valid = False
                            break
                    if valid:
                        s = new_s | {x}
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # 3. Try 2-for-1 swaps (Replace two smaller with one larger)
            # This is more expensive, so we do it less frequently or only if time permits
            # For now, include it in the loop but it's O(|S|^2 * N * |S|)
            lst = list(s)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a = lst[i]
                    b = lst[j]
                    pair_sum = a + b
                    for c in range(N, 0, -1):
                        if c in s:
                            continue
                        if c <= pair_sum:
                            continue
                        new_s = s - {a, b}
                        valid = True
                        for z in new_s:
                            if math.gcd(c, z) != 1:
                                valid = False
                                break
                        if valid:
                            s = new_s | {c}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue

            # 4. Try 1-for-2 swaps (Replace one larger with two smaller)
            # This allows filling gaps with multiple numbers
            for a in list(s):
                # Find two numbers b, c not in s, b < c, b+c > a, gcd(b,c)=1, gcd(b, rest)=1, gcd(c, rest)=1
                # To speed up, iterate b, then c
                for b in range(1, N + 1):
                    if b in s:
                        continue
                    for c in range(b + 1, N + 1):
                        if c in s:
                            continue
                        if b + c <= a:
                            continue
                        if math.gcd(b, c) != 1:
                            continue
                        new_s = s - {a}
                        valid = True
                        for z in new_s:
                            if math.gcd(b, z) != 1 or math.gcd(c, z) != 1:
                                valid = False
                                break
                        if valid:
                            s = new_s | {b, c}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        return s

    best_set = set()
    best_sum = 0
    
    # Strategy 1: Greedy Descending
    s1 = greedy_construct(sorted(candidates, reverse=True))
    s1 = local_search(s1)
    if sum(s1) > best_sum:
        best_sum = sum(s1)
        best_set = s1
        
    # Strategy 2: Greedy with random restarts
    # Use remaining time for a few random restarts to escape local optima
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Randomize the order slightly? 
            # A better heuristic: Sort by value, but with random tie-breaking doesn't help as values are unique.
            # Instead, try a different greedy: 
            # Sort by number of prime factors ascending? 
            # Or just shuffle and greedy?
            # Shuffled greedy is weak. 
            # Let's try Greedy on Descending, but skip the first few? No.
            
            # Let's try a "Prime Power" focused greedy
            # Prioritize numbers with high value / (number of prime factors)
            # Or just run local search on the previous result with a random perturbation?
            
            # Perturbation: Remove a random element, then re-run local search?
            # Or just run a fresh greedy with a slightly different order.
            # Let's try sorting by descending value, but with a random weight added?
            # weight = value + random.uniform(-10, 10)
            
            weighted_candidates = [(x, x + random.uniform(-10, 10)) for x in candidates]
            weighted_candidates.sort(key=lambda item: item[1], reverse=True)
            order = [x for x, _ in weighted_candidates]
            
            s_rand = greedy_construct(order)
            s_rand = local_search(s_rand)
            
            if sum(s_rand) > best_sum:
                best_sum = sum(s_rand)
                best_set = s_rand

    # Ensure we return a list of ints
    return list(best_set)
