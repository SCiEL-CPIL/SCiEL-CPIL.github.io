import numpy as np
from math import lcm, gcd
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    weights = [1.0 / x for x in nums]
    
    # Build conflict bitsets for fast checking
    # conflict[i] is a bitmask where bit j is set if nums[i] and nums[j] conflict (lcm <= 256)
    conflict = [0] * n
    for i in range(n):
        a = nums[i]
        bits = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            # lcm(a, b) = a * b // gcd(a, b)
            # Check if lcm <= 256
            # Equivalent to: a // gcd(a,b) <= 256 // b
            # This avoids large integer overflow potential (though Python handles big ints, it's faster)
            g = gcd(a, b)
            if (a // g) * b <= 256:
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
            # conflict[i] has bits for all j != i that conflict with i
            if sol & conflict[i]:
                return False
            x ^= lsb
        return True
    
    # Initial Solution 1: Greedy by ascending value (smallest first, highest weight)
    def make_initial_greedy():
        sol = 0
        for i in range(n):
            if (sol & conflict[i]) == 0:
                sol |= (1 << i)
        return sol
    
    # Initial Solution 2: All numbers in (128, 256]
    # If a, b > 128, then a*b > 128*128 = 16384.
    # lcm(a,b) = a*b/gcd(a,b).
    # For lcm <= 256, we need a*b/gcd(a,b) <= 256 => a*b <= 256*gcd(a,b).
    # Since gcd(a,b) <= min(a,b) <= 256, a*b <= 256*256 = 65536.
    # If a,b > 128, min(a,b) >= 129.
    # If they are coprime, lcm = a*b > 129*129 > 256.
    # If they share a factor, say d, a=da, b=db. d^2 ab <= 256? No, lcm = dab.
    # Actually, let's just check validity programmatically.
    # Numbers 129..256.
    # If a,b in 129..256, is lcm(a,b) > 256 always?
    # Counter-example?
    # a=144, b=128? 128 is not in 129..256.
    # a=192, b=128? No.
    # a=256, b=128? No.
    # a=144, b=180? gcd=36. lcm = 144*180/36 = 4*180 = 720 > 256.
    # a=129, b=130? coprime, lcm > 256.
    # It seems {129..256} is a valid independent set.
    # Let's verify.
    sol2 = 0
    for i in range(n):
        if nums[i] > 128:
            sol2 |= (1 << i)
    
    best_sol = make_initial_greedy()
    best_w = sol_weight(best_sol)
    
    if valid(sol2):
        w2 = sol_weight(sol2)
        if w2 > best_w:
            best_w = w2
            best_sol = sol2
    
    # Local Search
    def local_search(sol, w, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-12
            best_move = None  # (new_sol, gain)
            
            # 1. 1-Add: Try to add any element j not in sol
            for j in range(n):
                if sol & (1 << j):
                    continue
                if (sol & conflict[j]) == 0:
                    gain = weights[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (sol | (1 << j), gain)
            
            # 2. 1-Swap: Remove i, add j
            for i in range(n):
                if not (sol & (1 << i)):
                    continue
                base = sol ^ (1 << i)
                # Try to add any j not in base
                for j in range(n):
                    if base & (1 << j):
                        continue
                    if (base & conflict[j]) == 0:
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (base | (1 << j), gain)
            
            # 3. 2-Add: Add two elements j, k simultaneously
            # This can escape local optima where no single add is possible, but two are.
            # Complexity is O(n^2) per iteration, which is ~65k ops. Acceptable.
            if time.time() < deadline - 0.5:
                # Get list of indices not in sol
                not_in_sol = [j for j in range(n) if not (sol & (1 << j))]
                # Filter not_in_sol to those that don't conflict with sol
                compatible_with_sol = [j for j in not_in_sol if (sol & conflict[j]) == 0]
                
                for idx_j in range(len(compatible_with_sol)):
                    j = compatible_with_sol[idx_j]
                    for idx_k in range(idx_j + 1, len(compatible_with_sol)):
                        k = compatible_with_sol[idx_k]
                        # Check if j and k conflict
                        if (conflict[j] & (1 << k)) != 0:
                            continue
                        gain = weights[j] + weights[k]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (sol | (1 << j) | (1 << k), gain)
            
            # 4. 2-Swap: Remove i1, i2, add j1, j2
            # This is O(n^4) which is too slow (~400M ops).
            # We can limit this to "Remove one, add two" or "Remove two, add one" if beneficial?
            # "Remove two, add one": Gain = w(j) - w(i1) - w(i2). Since w(j) <= 1/2 and w(i) >= 1/256,
            # removing two small numbers to add one large number is usually bad for sum.
            # "Remove one, add two": Gain = w(j1)+w(j2) - w(i).
            # This is O(n^3) ~ 16M ops. Might be too slow per iteration if many iterations.
            # Let's stick to 1-Add, 1-Swap, 2-Add for now. 2-Add is the key improvement over standard hill climb.
            
            if best_move is not None:
                sol, w = best_move[0], w + best_gain
                improved = True
        
        return sol, w
    
    # Run local search on the best initial solution
    best_sol, best_w = local_search(best_sol, best_w, deadline)
    
    # Restart with different initializations if time permits
    # Use random shuffles to break symmetry or find different basins
    random.seed(42)
    
    # We can also try greedy by different orderings
    # E.g., descending value (largest first) - likely poor sum but maybe different structure?
    # Or random order.
    
    attempts = 0
    max_attempts = 20 # Limit restarts to avoid overhead
    while time.time() < deadline - 2.0 and attempts < max_attempts:
        attempts += 1
        
        # Random Greedy Init
        order = list(range(n))
        random.shuffle(order)
        sol = 0
        for i in order:
            if (sol & conflict[i]) == 0:
                sol |= (1 << i)
        
        w = sol_weight(sol)
        sol, w = local_search(sol, w, deadline)
        
        if w > best_w:
            best_w = w
            best_sol = sol
    
    # Final Polish: Try to remove one element and add the best possible set of compatible elements
    # This is a "1-Remove, k-Add" move.
    if time.time() < deadline - 1.0:
        polish_sol = best_sol
        polish_w = best_w
        
        # Iterate through elements in current solution
        in_sol_indices = [j for j in range(n) if polish_sol & (1 << j)]
        
        for i in in_sol_indices:
            if time.time() >= deadline - 0.5:
                break
            
            base = polish_sol ^ (1 << i)
            base_w = polish_w - weights[i]
            
            # Find all elements compatible with base
            # and not in base
            compatible = []
            for j in range(n):
                if base & (1 << j):
                    continue
                if (base & conflict[j]) == 0:
                    compatible.append(j)
            
            # Sort compatible by weight descending
            compatible.sort(key=lambda x: weights[x], reverse=True)
            
            # Try to add as many as possible from this sorted list (Greedy Add)
            # This is a local greedy fill on the base
            trial_sol = base
            trial_w = base_w
            for j in compatible:
                if (trial_sol & conflict[j]) == 0:
                    trial_sol |= (1 << j)
                    trial_w += weights[j]
            
            if trial_w > polish_w:
                polish_w = trial_w
                polish_sol = trial_sol
                # Update best if improved
                if polish_w > best_w:
                    best_w = polish_w
                    best_sol = polish_sol
    
    # Convert bitset back to list
    result = [nums[i] for i in range(n) if best_sol & (1 << i)]
    
    # Final Validation
    if len(result) > 0:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if lcm(result[i], result[j]) <= 256:
                    # Should not happen, but safety check
                    # If it does, return empty? No, must return nonempty.
                    # Fallback to a known valid set: {129..256}
                    return list(range(129, 257))
    
    return result
