import numpy as np
from math import lcm, gcd
import time
import sys
import random
from itertools import combinations

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
    
    # Local Search with 1-Add, 1-Swap, 2-Add, and 1-Remove-2-Add moves
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
                for j in range(n):
                    if base & (1 << j):
                        continue
                    if (base & conflict[j]) == 0:
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (base | (1 << j), gain)
            
            # 3. 2-Add: Add two elements j, k simultaneously
            if time.time() < deadline - 0.5:
                not_in_sol = [j for j in range(n) if not (sol & (1 << j))]
                compatible_with_sol = [j for j in not_in_sol if (sol & conflict[j]) == 0]
                
                for idx_j in range(len(compatible_with_sol)):
                    j = compatible_with_sol[idx_j]
                    for idx_k in range(idx_j + 1, len(compatible_with_sol)):
                        k = compatible_with_sol[idx_k]
                        if (conflict[j] & (1 << k)) != 0:
                            continue
                        gain = weights[j] + weights[k]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (sol | (1 << j) | (1 << k), gain)
            
            # 4. 1-Remove-2-Add: Remove one element i, add two elements j, k
            # This can find solutions where removing a low-weight element allows adding two higher-weight elements
            if time.time() < deadline - 1.0:
                in_sol = [i for i in range(n) if sol & (1 << i)]
                not_in_sol = [j for j in range(n) if not (sol & (1 << j))]
                
                for i in in_sol:
                    if time.time() >= deadline - 0.5:
                        break
                    base = sol ^ (1 << i)
                    # Find elements compatible with base
                    compatible = [j for j in not_in_sol if (base & conflict[j]) == 0]
                    
                    # Try all pairs from compatible
                    for idx_j in range(len(compatible)):
                        j = compatible[idx_j]
                        for idx_k in range(idx_j + 1, len(compatible)):
                            k = compatible[idx_k]
                            if (conflict[j] & (1 << k)) != 0:
                                continue
                            gain = weights[j] + weights[k] - weights[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (base | (1 << j) | (1 << k), gain)
            
            if best_move is not None:
                sol, w = best_move[0], w + best_gain
                improved = True
        
        return sol, w
    
    # Run local search on the best initial solution
    best_sol, best_w = local_search(best_sol, best_w, deadline)
    
    # Restart with different initializations if time permits
    random.seed(42)
    
    attempts = 0
    max_attempts = 20
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
    
    # Final Polish: 1-Remove, k-Add with greedy fill
    if time.time() < deadline - 1.0:
        polish_sol = best_sol
        polish_w = best_w
        
        in_sol_indices = [j for j in range(n) if polish_sol & (1 << j)]
        
        for i in in_sol_indices:
            if time.time() >= deadline - 0.5:
                break
            
            base = polish_sol ^ (1 << i)
            base_w = polish_w - weights[i]
            
            compatible = []
            for j in range(n):
                if base & (1 << j):
                    continue
                if (base & conflict[j]) == 0:
                    compatible.append(j)
            
            compatible.sort(key=lambda x: weights[x], reverse=True)
            
            trial_sol = base
            trial_w = base_w
            for j in compatible:
                if (trial_sol & conflict[j]) == 0:
                    trial_sol |= (1 << j)
                    trial_w += weights[j]
            
            if trial_w > polish_w:
                polish_w = trial_w
                polish_sol = trial_sol
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
                    return list(range(129, 257))
    
    return result
