"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # Sort descending for potential early termination
    
    # Dijkstra's algorithm on state = (current_sum, frozenset_of_used_divisors)
    # But that state space is too large. Instead, we use a different approach.
    # 
    # Since we want to minimize the number of divisors, we can use BFS by count.
    # State: (sum_so_far, frozenset of used divisors)
    # But the number of subsets is huge.
    #
    # Better approach: Since 5040 has only 60 divisors, and target is 4979,
    # we can use meet-in-the-middle or a smarter search.
    #
    # Actually, let's think about this differently. The maximum divisor is 5040.
    # 4979 is close to 5040. 5040 - 4979 = 61.
    # Can we represent 4979 as 5040 - 61? No, because we need sum of divisors, not difference.
    # 
    # Let's try: can we use 5040 itself? 5040 > 4979, so no.
    # Largest divisor <= 4979: let's find it.
    # 5040/2 = 2520, 5040/1 = 5040 (too big)
    # Divisors of 5040: 1,2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,30,35,36,40,42,45,48,56,60,63,70,72,80,84,90,105,112,120,126,140,144,168,180,210,224,240,252,280,315,336,360,420,504,560,630,720,840,1008,1260,1512,1680,2520,5040
    # 
    # Largest <= 4979: 2520
    # 4979 - 2520 = 2459
    # Largest divisor <= 2459: 1680
    # 2459 - 1680 = 779
    # Largest divisor <= 779: 720
    # 779 - 720 = 59
    # Largest divisor <= 59: 56
    # 59 - 56 = 3
    # 3 is a divisor!
    # So: 2520 + 1680 + 720 + 56 + 3 = 4979
    # That's 5 divisors.
    
    # Let's verify: 2520 + 1680 = 4200, + 720 = 4920, + 56 = 4976, + 3 = 4979. Yes!
    # All are distinct divisors of 5040.
    # 
    # Can we do it in fewer? 4 divisors?
    # Max sum with 4 divisors: 2520 + 1680 + 720 + 504 = 5424 > 4979, so possible in principle.
    # But we need exact sum. Let's check if 4 is possible.
    # 
    # For now, let's return this 5-divisor solution and see if we can improve.
    # 
    # Actually, let's use a proper search to find the minimum.
    # 
    # Since the problem is small (60 divisors), we can use a BFS/DFS approach.
    # State: (current_sum, used_divisors_set)
    # We want to reach target with minimum number of divisors.
    
    # Precompute divisors
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs_list = sorted(divs, reverse=True)
    
    # BFS by number of divisors
    # Level k: all sums achievable with exactly k distinct divisors
    # We use a set of (sum, frozenset) but that's too memory intensive.
    # Instead, for each level, we track all achievable sums and one way to achieve them.
    
    # Better: use a dictionary mapping sum -> (count, list_of_divisors)
    # But we need to avoid using the same divisor twice.
    
    # Let's use a different approach: since 5040 has only 60 divisors, and we're looking for small k,
    # we can do DFS with pruning.
    
    best_solution = None
    best_count = float('inf')
    
    # Sort divisors in descending order for better pruning
    divs_sorted = sorted(divs, reverse=True)
    
    def dfs(remaining, start_idx, current_divs, count):
        nonlocal best_solution, best_count
        if time.time() > deadline:
            return
        if remaining == 0:
            if count < best_count:
                best_count = count
                best_solution = list(current_divs)
            return
        if remaining < 0:
            return
        if count >= best_count:
            return
        # Pruning: if even using the largest available divisors we can't reach remaining
        # (this is a weak bound, but helps)
        for i in range(start_idx, len(divs_sorted)):
            d = divs_sorted[i]
            if d > remaining:
                continue
            # Additional pruning: sum of all remaining divisors from i onwards must be >= remaining
            # This is expensive to compute, so skip for now
            current_divs.append(d)
            dfs(remaining - d, i + 1, current_divs, count + 1)
            current_divs.pop()
    
    dfs(target, 0, [], 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: return the known 5-divisor solution
    return [2520, 1680, 720, 56, 3]
# EVOLVE-BLOCK-END
