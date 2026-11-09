import time
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from math import isqrt, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s kill

    N = 256
    # 1. Find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Map numbers 2..256 to indices 0..254
    nums = list(range(2, N + 1))
    n_vars = len(nums)
    num_to_idx = {n: i for i, n in enumerate(nums)}
    
    # 3. Build constraint matrix
    n_primes = len(primes)
    A = np.zeros((n_primes, n_vars), dtype=int)
    
    for v, n in enumerate(nums):
        for p_idx, p in enumerate(primes):
            if n % p == 0:
                A[p_idx, v] = 1
    
    # 4. Define ILP
    c = np.array([-n for n in nums], dtype=float)
    constraint = LinearConstraint(A, lb=0, ub=1)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars, dtype=int)
    
    # Solve
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        selected = [1]
        for n in sorted(nums, reverse=True):
            if all(gcd(n, s) == 1 for s in selected):
                selected.append(n)
        return [int(x) for x in selected]
    
    x = res.x
    selected = [1]
    for i, val in enumerate(x):
        if val > 0.5:
            selected.append(nums[i])
    
    # Precompute conflict bitmask for each number (among nums 2..256)
    # conflict_mask[i] is a bitmask where bit j is set if nums[i] and nums[j] share a gcd > 1
    # We use a set of indices for the selected set for fast lookup
    # Actually, let's precompute a matrix: conflict[i][j] = True if gcd(nums[i], nums[j]) > 1
    # For speed with 255 elements, use a list of sets
    conflict_sets = [set() for _ in range(n_vars)]
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if gcd(nums[i], nums[j]) > 1:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
    
    # Helper: check if a number (by index) is coprime to all selected (by index set)
    def is_compatible(idx, sel_set):
        """Check if nums[idx] is coprime to all elements in sel_set (set of indices)"""
        for s in sel_set:
            if s in conflict_sets[idx]:
                return False
        return True
    
    # Convert selected to index set (excluding 1 which has no index in nums)
    sel_set = set()
    for s in selected:
        if s != 1:
            sel_set.add(num_to_idx[s])
    
    unselected = set(range(n_vars)) - sel_set
    
    # Local search with 1-for-1 swaps, additions, and 2-for-1 swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding each unselected number if it's coprime to all selected
        for idx in sorted(unselected, key=lambda i: nums[i], reverse=True):
            if time.time() >= deadline:
                break
            if is_compatible(idx, sel_set):
                sel_set.add(idx)
                unselected.discard(idx)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove one selected, add one unselected
        sel_list = sorted(sel_set, key=lambda i: nums[i], reverse=True)
        for rem_idx in sel_list:
            if time.time() >= deadline:
                break
            for add_idx in sorted(unselected, key=lambda i: nums[i], reverse=True):
                if nums[add_idx] <= nums[rem_idx]:
                    continue
                # Check if add_idx is coprime to all selected except rem_idx
                if is_compatible(add_idx, sel_set - {rem_idx}):
                    sel_set.discard(rem_idx)
                    sel_set.add(add_idx)
                    unselected.discard(add_idx)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-for-1 swaps: remove two selected, add one unselected
        # This is O(n^2 * m) which could be slow, so be selective
        # Only try if the sum of the two removed < the one added
        sel_list = sorted(sel_set, key=lambda i: nums[i])
        # Sort unselected by value descending
        unsel_sorted = sorted(unselected, key=lambda i: nums[i], reverse=True)
        
        found_2for1 = False
        for add_idx in unsel_sorted:
            if time.time() >= deadline:
                break
            add_val = nums[add_idx]
            # Find pairs of selected whose sum < add_val
            for i in range(len(sel_list)):
                if time.time() >= deadline:
                    break
                for j in range(i+1, len(sel_list)):
                    rem1, rem2 = sel_list[i], sel_list[j]
                    if nums[rem1] + nums[rem2] >= add_val:
                        continue
                    # Check if add_idx is coprime to all selected except rem1, rem2
                    temp_sel = sel_set - {rem1, rem2}
                    if is_compatible(add_idx, temp_sel):
                        sel_set.discard(rem1)
                        sel_set.discard(rem2)
                        sel_set.add(add_idx)
                        unselected.discard(add_idx)
                        improved = True
                        found_2for1 = True
                        break
                if found_2for1:
                    break
            if found_2for1:
                break
    
    # Reconstruct result
    selected = [1]
    for idx in sorted(sel_set):
        selected.append(nums[idx])
    
    return [int(s) for s in selected]
