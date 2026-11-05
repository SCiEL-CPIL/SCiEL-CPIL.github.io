import numpy as np
from math import lcm, gcd
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    weights = [1.0 / x for x in nums]

    # Build conflict bitsets
    # conflict[i] is a bitmask where bit j is set if nums[i] and nums[j] conflict
    # conflict[i] does NOT include bit i itself
    conflict = [0] * n
    for i in range(n):
        a = nums[i]
        bits = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            # lcm(a, b) = a * b // gcd(a, b)
            # Optimization: if a > 256, no conflicts (not possible here)
            # if a * b // gcd(a,b) <= 256 -> conflict
            # Since a,b >= 2, max product 256*256.
            # Simple check is fine for n=255.
            g = gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                bits |= (1 << j)
        conflict[i] = bits

    def sol_weight(sol):
        s = 0.0
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            s += weights[i]
            x ^= lsb
        return s

    def valid(sol):
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            # Check if i conflicts with any other in sol
            # conflict[i] contains indices j != i that conflict with i
            # If sol & conflict[i] is non-zero, there is a conflict
            if sol & conflict[i]:
                return False
            x ^= lsb
        return True

    def make_greedy(ordered_indices):
        sol = 0
        for i in ordered_indices:
            if (sol & conflict[i]) == 0:
                sol |= (1 << i)
        return sol

    def local_search(sol, w, deadline):
        improved = True
        # Precompute list of indices for iteration to avoid range overhead in tight loop?
        # n=255, so range(n) is fine.
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-12
            best_move = None  # (new_sol, gain)

            # 1-add: try adding any node not in sol
            # If sol is large, this is expensive. But n=255.
            for j in range(n):
                if sol & (1 << j):
                    continue
                if (sol & conflict[j]) == 0:
                    gain = weights[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (sol | (1 << j), gain)

            # 1-swap (remove i, add j)
            # Iterate over i in sol
            x = sol
            while x:
                lsb_i = x & (-x)
                i = lsb_i.bit_length() - 1
                x ^= lsb_i
                
                base = sol ^ (1 << i)
                # Try adding j not in sol
                # Optimization: if we removed i, maybe we can add a j that was conflicting with i
                # but not with base.
                for j in range(n):
                    if sol & (1 << j):
                        continue
                    if (base & conflict[j]) == 0:
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (base | (1 << j), gain)
                
                if time.time() > deadline:
                    break
                if best_move and best_gain > 0.01: # Heuristic early exit? No, keep strict.
                    pass

            if best_move is not None:
                sol, w = best_move[0], w + best_gain
                improved = True

        return sol, w

    # Initial solutions
    best_sol = 0
    best_w = 0.0

    # 1. Standard greedy (ascending)
    sol1 = make_greedy(range(n))
    w1 = sol_weight(sol1)
    if w1 > best_w:
        best_w = w1
        best_sol = sol1

    # 2. Greedy (descending) - picks large numbers first, might miss small high-weight ones
    sol2 = make_greedy(range(n - 1, -1, -1))
    w2 = sol_weight(sol2)
    if w2 > best_w:
        best_w = w2
        best_sol = sol2

    # 3. Randomized Greedy restarts
    import random
    random.seed(42)
    
    # Run local search on initial bests first
    if time.time() < deadline:
        best_sol, best_w = local_search(best_sol, best_w, deadline)

    # Multiple restarts with random orders
    # We have ~80s. Local search is the expensive part.
    # Let's do a few random restarts.
    max_restarts = 20
    for _ in range(max_restarts):
        if time.time() > deadline - 5: # Leave 5s for final check
            break
            
        order = list(range(n))
        random.shuffle(order)
        
        # Weighted random? Or just uniform?
        # Uniform shuffle is fine.
        
        sol = make_greedy(order)
        w = sol_weight(sol)
        
        # Local search
        sol, w = local_search(sol, w, deadline)
        
        if w > best_w:
            best_w = w
            best_sol = sol

    # Convert back to list of numbers
    result = []
    x = best_sol
    while x:
        lsb = x & (-x)
        i = lsb.bit_length() - 1
        result.append(nums[i])
        x ^= lsb
        
    # Final verification
    if not valid(best_sol):
        # Fallback to a known valid set if something went wrong
        # e.g. just [256] or greedy
        return make_greedy(range(n)) and [nums[i] for i in range(n) if best_sol & (1<<i)]
        
    return result

# Note: The valid check in the return logic above is slightly redundant if local_search is correct,
# but good for safety. The line `return make_greedy...` is a bit clunky, let's clean it up.

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    weights = [1.0 / x for x in nums]

    conflict = [0] * n
    for i in range(n):
        a = nums[i]
        bits = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            g = gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                bits |= (1 << j)
        conflict[i] = bits

    def sol_weight(sol):
        s = 0.0
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            s += weights[i]
            x ^= lsb
        return s

    def valid(sol):
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            if sol & conflict[i]:
                return False
            x ^= lsb
        return True

    def make_greedy(ordered_indices):
        sol = 0
        for i in ordered_indices:
            if (sol & conflict[i]) == 0:
                sol |= (1 << i)
        return sol

    def local_search(sol, w, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-12
            best_move = None

            # 1-add
            for j in range(n):
                if sol & (1 << j):
                    continue
                if (sol & conflict[j]) == 0:
                    gain = weights[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (sol | (1 << j), gain)

            # 1-swap
            x = sol
            while x:
                lsb_i = x & (-x)
                i = lsb_i.bit_length() - 1
                x ^= lsb_i
                
                base = sol ^ (1 << i)
                for j in range(n):
                    if sol & (1 << j):
                        continue
                    if (base & conflict[j]) == 0:
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (base | (1 << j), gain)
                
                if time.time() > deadline:
                    break

            if best_move is not None:
                sol, w = best_move[0], w + best_gain
                improved = True

        return sol, w

    best_sol = 0
    best_w = 0.0

    # Initial: Ascending greedy
    sol1 = make_greedy(range(n))
    w1 = sol_weight(sol1)
    if w1 > best_w:
        best_w = w1
        best_sol = sol1

    # Initial: Descending greedy
    sol2 = make_greedy(range(n - 1, -1, -1))
    w2 = sol_weight(sol2)
    if w2 > best_w:
        best_w = w2
        best_sol = sol2

    # Local search on best initial
    if time.time() < deadline:
        best_sol, best_w = local_search(best_sol, best_w, deadline)

    # Random restarts
    import random
    random.seed(42)
    
    while time.time() < deadline - 5:
        order = list(range(n))
        random.shuffle(order)
        sol = make_greedy(order)
        w = sol_weight(sol)
        
        sol, w = local_search(sol, w, deadline)
        
        if w > best_w:
            best_w = w
            best_sol = sol

    result = []
    x = best_sol
    while x:
        lsb = x & (-x)
        i = lsb.bit_length() - 1
        result.append(nums[i])
        x ^= lsb
        
    return result
