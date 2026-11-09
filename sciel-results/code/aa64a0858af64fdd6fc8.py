import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = 85.0  # 5 seconds safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort in descending order to prioritize larger divisors
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute suffix sums for pruning: suffix_sum[i] is sum of divisors[i:]
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # Meet-in-the-middle approach for better performance
    # Split divisors into two halves
    mid = m // 2
    left_divs = divisors[:mid]
    right_divs = divisors[mid:]
    
    # Generate all subset sums for left half, tracking which subsets give each sum
    # and the number of elements
    from itertools import combinations
    
    left_sums = {}  # sum -> (count, indices)
    for r in range(len(left_divs) + 1):
        for combo in combinations(range(len(left_divs)), r):
            s = sum(left_divs[i] for i in combo)
            if s not in left_sums:
                left_sums[s] = (r, combo)
    
    # Generate all subset sums for right half
    right_sums = {}  # sum -> (count, indices)
    for r in range(len(right_divs) + 1):
        for combo in combinations(range(len(right_divs)), r):
            s = sum(right_divs[i] for i in combo)
            if s not in right_sums:
                right_sums[s] = (r, combo)
    
    # Find minimum total count where left_sum + right_sum = target
    best_count = float('inf')
    best_solution = None
    
    for ls, (lc, lcombo) in left_sums.items():
        rs_needed = target - ls
        if rs_needed in right_sums:
            rc, rcombo = right_sums[rs_needed]
            total_count = lc + rc
            if total_count < best_count:
                best_count = total_count
                best_solution = ([left_divs[i] for i in lcombo] + [right_divs[i] for i in rcombo], total_count)
    
    if best_solution is not None:
        return best_solution[0]
    
    # Fallback to BFS if MITM fails (shouldn't happen)
    q = deque()
    q.append((target, -1, []))
    
    while q:
        if time.time() > deadline:
            pass
            
        remaining, last_idx, path = q.popleft()
        
        if remaining == 0:
            return path
        
        for i in range(last_idx + 1, m):
            d = divisors[i]
            if d > remaining:
                continue
            
            new_remaining = remaining - d
            new_path = path + [d]
            
            if new_remaining > suffix_sum[i + 1]:
                continue
            
            q.append((new_remaining, i, new_path))
            
    return []
