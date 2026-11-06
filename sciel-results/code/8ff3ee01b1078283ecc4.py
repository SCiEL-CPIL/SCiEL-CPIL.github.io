import math
from functools import reduce
import time
import random

def construct():
    # Calculate LCM of 1..24
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = reduce(lcm, range(1, 25))
    weights = {i: L // i for i in range(1, 25)}
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def is_valid(S):
        # S is a set of integers
        if not S:
            return True
        S_list = list(S)
        for x in S_list:
            w_x = weights[x]
            others = [weights[y] for y in S_list if y != x]
            if not others:
                continue
            
            # Check if w_x is a subset sum of others
            # Use DFS with pruning
            others.sort(reverse=True)
            n = len(others)
            suffix_sums = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + others[i]
            
            if suffix_sums[0] < w_x:
                # Sum of all others is less than w_x, so w_x cannot be formed
                # But we need to check if w_x is formed. If total < w_x, it's impossible.
                # Wait, if total < w_x, then w_x is NOT a subset sum. So this x is "safe".
                # We return False only if w_x IS a subset sum.
                # So if total < w_x, we continue to next x.
                continue
                
            # Memoization for DFS
            # State: (index, remaining)
            # Since w_x can be large, we use a set of visited states?
            # Or just simple recursion with pruning.
            # To avoid deep recursion limits, we can use iterative or increase limit.
            # But N=24 is small.
            
            visited = set()
            def dfs(i, rem):
                if rem == 0:
                    return True
                if i >= n:
                    return False
                if rem < 0:
                    return False
                if suffix_sums[i] < rem:
                    return False
                
                state = (i, rem)
                if state in visited:
                    return False
                visited.add(state)
                
                # Option 1: Take others[i]
                if dfs(i + 1, rem - others[i]):
                    return True
                # Option 2: Don't take others[i]
                if dfs(i + 1, rem):
                    return True
                return False
            
            if dfs(0, w_x):
                return False
        return True

    def get_neighbors(S):
        S_list = list(S)
        neighbors = []
        
        # Additions
        for y in range(1, 25):
            if y not in S:
                new_S = S.union([y])
                if is_valid(new_S):
                    neighbors.append(new_S)
        
        # Swaps (remove x, add y)
        for x in S_list:
            for y in range(1, 25):
                if y not in S:
                    new_S = S.difference([x]).union([y])
                    if is_valid(new_S):
                        neighbors.append(new_S)
                        
        return neighbors

    best_set = set()
    best_size = 0
    
    # Multiple Greedy Starts
    start_strategies = [
        list(range(24, 0, -1)), # Largest to smallest
        list(range(1, 25)),     # Smallest to largest
        sorted(range(1, 25), key=lambda x: random.random())
    ]
    
    for _ in range(5):
        start_strategies.append(sorted(range(1, 25), key=lambda x: random.random()))
        
    for order in start_strategies:
        if time.time() > deadline:
            break
            
        reachable = {0}
        selected = []
        
        for x in order:
            w_x = weights[x]
            if w_x in reachable:
                continue
            else:
                selected.append(x)
                new_sums = set()
                for s in reachable:
                    ns = s + w_x
                    new_sums.add(ns)
                reachable.update(new_sums)
                
        current_set = set(selected)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            neighbors = get_neighbors(current_set)
            
            best_neighbor = None
            best_neighbor_size = current_size
            
            for n in neighbors:
                if len(n) > best_neighbor_size:
                    best_neighbor_size = len(n)
                    best_neighbor = n
                    
            if best_neighbor is not None:
                current_set = best_neighbor
                current_size = best_neighbor_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
            else:
                # If stuck, try a random perturbation?
                # For now, break to try next start
                break
                
    return list(best_set)
