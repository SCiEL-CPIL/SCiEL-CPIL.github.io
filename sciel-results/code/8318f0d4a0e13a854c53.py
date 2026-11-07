"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill limit
    
    N = 256
    candidates = list(range(1, N + 1))
    
    # Pre-compute conflicts: i and j conflict if gcd(i, j) > 1
    # To save memory/time, we can check on the fly, but pre-computing adjacency 
    # or a simple compatibility matrix might help. 
    # Given N=256, a 256x256 matrix is small (65k entries).
    # Let's use a bitset or a list of sets for fast lookup.
    # conflict[i] = set of j > i such that gcd(i,j) > 1
    # Actually, we just need to check if a new candidate is coprime to all selected.
    
    # Optimization: 
    # 1. Sort candidates by value descending to prioritize high weights in search tree.
    # 2. Use branch and bound.
    #    State: (index, current_sum, selected_list)
    #    Bound: current_sum + sum of all remaining candidates that are coprime to all selected?
    #           No, that's too optimistic because remaining candidates might conflict with each other.
    #           A tighter bound: current_sum + sum of remaining candidates (ignoring conflicts among themselves).
    #           This is a valid upper bound. If this bound <= best_sum, prune.
    
    # However, simple B&B on 256 nodes might be too slow if not careful.
    # The "conflict" structure is specific: primes are independent of each other.
    # Composites share factors.
    
    # Let's try a different angle: 
    # The set of primes is a valid solution. Sum of primes <= 256 is ~1060.
    # Can we do better? 
    # Replacing a prime p with a composite c is only possible if c is coprime to all other selected.
    # Usually, composites are "heavier" in conflict. 
    # Example: 255 (3*5*17) is coprime to 2, 7, 11, etc.
    # 255 is large. Is it better than 251 (prime)? 
    # If we include 255, we cannot include any multiple of 3, 5, or 17.
    # Primes to lose: 3, 5, 17, 19? No, 19 is prime. 
    # Multiples of 3: 3, 9, 15, ... 255.
    # If we pick 255, we lose all primes that are 3, 5, 17. 
    # Sum lost: 3+5+17 = 25. Gain: 255. Net gain +230? 
    # Wait, we also lose composites that were coprime to 255 but coprime to 3,5,17? 
    # No, the base set was primes. So we only compare against primes.
    # If we start with all primes, and try to swap subsets.
    
    # Strategy: Local Search / Hill Climbing starting from All Primes.
    # 1. Start with S = {1} U {primes <= 256}.
    # 2. Compute sum.
    # 3. Try to add any non-selected number x if it is coprime to all in S.
    # 4. Try to swap: remove subset R from S, add subset A not in S, if valid and sum increases.
    #    Specifically, try adding one x. If conflict, try removing minimal conflicting subset.
    
    # This is a Max Weight Independent Set problem.
    # Given N=256, exact solution via B&B is feasible if pruned well.
    # Let's implement a robust B&B.
    
    # Pre-process:
    # Sort indices by value descending.
    # idx_sorted = sorted(range(1, N+1), key=lambda x: x, reverse=True)
    # values = [idx for idx in idx_sorted]
    
    # Conflict check:
    # coprime_cache = {}
    # def is_coprime(a, b):
    #     if (a,b) in coprime_cache: return coprime_cache[(a,b)]
    #     res = gcd(a,b) == 1
    #     coprime_cache[(a,b)] = res
    #     return res
    
    # To make B&B faster, we can precompute for each number, the list of numbers it conflicts with.
    # conflicts = [set() for _ in range(N+1)]
    # for i in range(1, N+1):
    #     for j in range(i+1, N+1):
    #         if gcd(i,j) > 1:
    #             conflicts[i].add(j)
    #             conflicts[j].add(i)
    
    # B&B State:
    # best_sum = 0
    # best_set = []
    
    # def search(idx, current_sum, selected):
    #     nonlocal best_sum, best_set
    #     if time.time() > deadline:
    #         return
    #     if idx == len(values):
    #         if current_sum > best_sum:
    #             best_sum = current_sum
    #             best_set = list(selected)
    #         return
    #     
    #     # Upper Bound: current_sum + sum(values[idx:])
    #     # This is weak. Better bound: current_sum + sum of values[idx:] that are coprime to all in selected?
    #     # Even better: current_sum + sum of values[idx:] that are coprime to all in selected AND form an independent set among themselves?
    #     # Too expensive.
    #     # Let's use: current_sum + sum(values[idx:])
    #     if current_sum + suffix_sum[idx] <= best_sum:
    #         return
    #         
    #     val = values[idx]
    #     # Option 1: Include val
    #     if all(gcd(val, s) == 1 for s in selected):
    #         selected.append(val)
    #         search(idx+1, current_sum + val, selected)
    #         selected.pop()
    #         
    #     # Option 2: Exclude val
    #     search(idx+1, current_sum, selected)
    
    # Problem: Depth 256, branching factor 2. 2^256 is huge. Pruning is key.
    # The bound `current_sum + suffix_sum[idx]` is very loose.
    # We need a tighter bound.
    # A common heuristic for MWIS:
    # If we pick val, we can only pick neighbors that are NOT in conflicts[val].
    # So the bound after picking val is current_sum + val + sum(values[idx+1:] that are not in conflicts[val] and coprime to selected).
    
    # Let's implement this tighter bound.
    
    values = list(range(N, 0, -1))
    n_vals = len(values)
    
    # Precompute suffix sums of values
    suffix_sum = [0] * (n_vals + 1)
    for i in range(n_vals - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + values[i]
    
    # Precompute conflict masks? 
    # Since N is small, we can just check gcd.
    # But checking "coprime to all in selected" is O(k) per step.
    
    best_sum = 0
    best_set = []
    
    # To speed up "coprime to all in selected", we can maintain a set of selected primes?
    # No, selected numbers are arbitrary.
    
    # Let's try a different search order or heuristic.
    # Greedy construction is fast but suboptimal.
    # B&B with good bounds is the way.
    
    # Optimization: 
    # If we include a number v, we exclude all its neighbors.
    # The bound can be improved by calculating the maximum weight independent set of the remaining graph 
    # restricted to non-neighbors of v. This is recursive and expensive.
    # Approximation: sum of weights of non-neighbors of v in the remaining list.
    
    def search(idx, current_sum, selected):
        nonlocal best_sum, best_set
        if time.time() > deadline:
            return
        
        if idx == n_vals:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(selected)
            return
            
        # Bound 1: Simple suffix sum
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        v = values[idx]
        
        # Check if v can be added
        can_add = True
        for s in selected:
            if gcd(v, s) != 1:
                can_add = False
                break
                
        if can_add:
            # Branch 1: Add v
            selected.append(v)
            
            # Calculate tighter bound for this branch
            # Sum of remaining values that are coprime to v AND coprime to selected
            # We can just use suffix_sum[idx+1] as a loose bound for now, 
            # or calculate sum of values[idx+1:] that are coprime to v.
            # Let's try to calculate a slightly tighter bound:
            # bound_add = current_sum + v + sum(x for x in values[idx+1:] if gcd(x, v) == 1)
            # Note: This ignores conflicts among the remaining x's, so it's a valid upper bound.
            # This calculation is O(N) per node. 256 * 2^depth is too slow if depth is large.
            # But pruning should keep depth small or branches few.
            
            # Let's just use the simple bound first and see if it's fast enough.
            # If not, we optimize.
            search(idx + 1, current_sum + v, selected)
            selected.pop()
            
        # Branch 2: Don't add v
        search(idx + 1, current_sum, selected)
        
    # Initial call
    # We can start with a greedy solution to set a high initial best_sum
    # Greedy: Sort by value desc, add if coprime to all current.
    greedy_set = []
    greedy_sum = 0
    for v in values:
        if all(gcd(v, s) == 1 for s in greedy_set):
            greedy_set.append(v)
            greedy_sum += v
    best_sum = greedy_sum
    best_set = list(greedy_set)
    
    # Run B&B
    # To avoid recursion limit issues, we might need sys.setrecursionlimit, but 256 is fine.
    # However, the number of nodes visited could be large.
    # Let's add a check: if time is running out, return best found.
    
    search(0, 0, [])
    
    # Ensure we return a valid list of ints
    result = best_set
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
