"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    divisors.sort(reverse=True)
    # Only keep divisors that could be part of a sum to target
    divisors = [d for d in divisors if d <= target]
    m = len(divisors)
    
    # For each k from 1 upward, check if any k-combination sums to target
    for k in range(1, min(20, m + 1)):
        if time.time() > deadline:
            break
        # Feasibility checks
        if sum(divisors[:k]) < target:
            continue  # Even largest k don't reach target
        if sum(divisors[-k:]) > target:
            break  # Even smallest k exceed target, no larger k will work either
        
        if k <= 4:
            # Direct combinations for small k
            for combo in combinations(divisors, k):
                if sum(combo) == target:
                    return list(combo)
        elif k <= 6:
            # Meet in the middle: split into two groups
            # Group A: first k//2 elements, Group B: rest
            # We need to pick i from A and k-i from B for all valid i
            half = k // 2
            # Enumerate all subsets of size j from divisors
            # Use meet-in-the-middle: split divisors into two halves
            mid = m // 2
            left = divisors[:mid]
            right = divisors[mid:]
            
            # For each possible split (i from left, k-i from right)
            found = False
            for i in range(max(0, k - len(right)), min(k, len(left)) + 1):
                j = k - i
                if i < 0 or j < 0:
                    continue
                # Enumerate all i-subsets of left and j-subsets of right
                # Build sum->indices map for one side
                if i == 0:
                    left_sums = {0: []}
                else:
                    left_sums = defaultdict(list)
                    for combo in combinations(left, i):
                        left_sums[sum(combo)].append(combo)
                
                if j == 0:
                    right_sums = {0: []}
                else:
                    right_sums = defaultdict(list)
                    for combo in combinations(right, j):
                        right_sums[sum(combo)].append(combo)
                
                # Find pairs that sum to target
                for ls, left_combos in left_sums.items():
                    needed = target - ls
                    if needed in right_sums:
                        for lc in left_combos:
                            for rc in right_sums[needed]:
                                # Check no overlap (they're from disjoint sets, so OK)
                                result = list(lc) + list(rc)
                                if len(set(result)) == k and sum(result) == target:
                                    return result
                        found = True
                        break
                if found:
                    break
        else:
            # For k >= 7, use a different approach:
            # Greedy with backtracking or use the fact that 5040 has many small divisors
            # Try a DFS with pruning
            def dfs(start, remaining, selected, depth):
                if time.time() > deadline:
                    return None
                if depth == 0:
                    if remaining == 0:
                        return list(selected)
                    return None
                # Pruning: even taking all remaining largest can't reach
                max_possible = sum(divisors[start:start+depth]) if start + depth <= m else sum(divisors[start:])
                if max_possible < remaining:
                    return None
                # Pruning: even taking smallest can exceed
                min_possible = sum(divisors[m-depth:m]) if m - depth >= 0 else sum(divisors[0:])
                if min_possible > remaining:
                    return None
                
                for idx in range(start, m - depth + 1):
                    d = divisors[idx]
                    if d > remaining:
                        continue
                    selected.append(d)
                    result = dfs(idx + 1, remaining - d, selected, depth - 1)
                    if result is not None:
                        return result
                    selected.pop()
                return None
            
            result = dfs(0, target, [], k)
            if result is not None:
                return result
    
    # Fallback
    return divisors[:1]

# EVOLVE-BLOCK-END
