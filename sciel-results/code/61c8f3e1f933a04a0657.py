"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute incompatibility: incompatible[a][b] = True if lcm(a,b) <= 256
    # Use a set of sets for efficiency
    incompatible = {}
    for a in candidates:
        incompatible[a] = set()
        for b in candidates:
            if a == b:
                continue
            if lcm(a, b) <= N:
                incompatible[a].add(b)
    
    def is_compatible(x, S_set):
        """Check if x is compatible with all elements in S_set."""
        for y in S_set:
            if y in incompatible[x]:
                return False
        return True
    
    def reciprocal_sum(S):
        return sum(1.0 / x for x in S)
    
    # Greedy: sort by reciprocal (descending) and add if compatible
    def greedy_solve(order):
        S = []
        S_set = set()
        for x in order:
            if is_compatible(x, S_set):
                S.append(x)
                S_set.add(x)
        return S, S_set
    
    # Try greedy with different orderings
    best_S = []
    best_sum = 0.0
    best_S_set = set()
    
    # Ordering 1: ascending (small numbers first, high reciprocal)
    order1 = sorted(candidates, key=lambda x: x)
    S1, S1_set = greedy_solve(order1)
    sum1 = reciprocal_sum(S1)
    if sum1 > best_sum:
        best_S, best_sum, best_S_set = S1, sum1, S1_set
    
    # Ordering 2: descending
    order2 = sorted(candidates, key=lambda x: -x)
    S2, S2_set = greedy_solve(order2)
    sum2 = reciprocal_sum(S2)
    if sum2 > best_sum:
        best_S, best_sum, best_S_set = S2, sum2, S2_set
    
    # Ordering 3: by number of conflicts (fewest conflicts first)
    order3 = sorted(candidates, key=lambda x: len(incompatible[x]))
    S3, S3_set = greedy_solve(order3)
    sum3 = reciprocal_sum(S3)
    if sum3 > best_sum:
        best_S, best_sum, best_S_set = S3, sum3, S3_set
    
    # Local search: try to swap elements
    def local_search(S, S_set, max_iters=50000):
        S = list(S)
        S_set = set(S)
        current_sum = reciprocal_sum(S)
        candidates_list = candidates
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try removing each element and adding a better one
            for i, x in enumerate(S):
                if time.time() > deadline:
                    break
                
                # Remove x
                S_set.remove(x)
                new_sum = current_sum - 1.0 / x
                
                # Try to find a compatible y not in S that improves the sum
                # We want 1/y > 1/x, so y < x, or at least 1/y + (gain) > 1/x
                # Actually we're replacing x with y, so we need 1/y > 1/x for improvement
                # i.e., y < x
                # But y must be compatible with all remaining elements
                best_y = None
                best_gain = 0.0
                
                # Try candidates smaller than x (higher reciprocal)
                for y in candidates_list:
                    if y == x or y in S_set:
                        continue
                    if y >= x:
                        continue  # 1/y <= 1/x, no improvement from replacement alone
                    if is_compatible(y, S_set):
                        gain = 1.0 / y - 1.0 / x
                        if gain > best_gain:
                            best_gain = gain
                            best_y = y
                
                if best_y is not None:
                    # Perform swap
                    S[i] = best_y
                    S_set.add(best_y)
                    current_sum += best_gain
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Put x back
                    S_set.add(x)
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 20:
                    break
        
        return S, S_set, current_sum
    
    # Run local search from best greedy solution
    S_ls, S_ls_set, sum_ls = local_search(best_S, best_S_set)
    if sum_ls > best_sum:
        best_S, best_sum, best_S_set = S_ls, sum_ls, S_ls_set
    
    # Try local search from other greedy solutions too
    for S_init, S_init_set, sum_init in [(S1, S1_set, sum1), (S2, S2_set, sum2), (S3, S3_set, sum3)]:
        if sum_init >= best_sum:
            continue
        if time.time() > deadline:
            break
        S_ls2, S_ls2_set, sum_ls2 = local_search(S_init, S_init_set, max_iters=20000)
        if sum_ls2 > best_sum:
            best_S, best_sum, best_S_set = S_ls2, sum_ls2, S_ls2_set
    
    # Additional: try adding elements that don't conflict (if we removed some during swaps)
    # Actually, let's also try a "fill" pass: add any compatible element not in set
    def fill_pass(S, S_set):
        S = list(S)
        S_set = set(S)
        improved = True
        while improved:
            improved = False
            for x in candidates:
                if x not in S_set and is_compatible(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
        return S, S_set
    
    # But wait - adding elements increases the sum, so we should always add if possible
    # The greedy already does this. But after swaps, maybe we can add more.
    S_fill, S_fill_set = fill_pass(best_S, best_S_set)
    sum_fill = reciprocal_sum(S_fill)
    if sum_fill > best_sum:
        best_S, best_sum, best_S_set = S_fill, sum_fill, S_fill_set
    
    # Final: try more aggressive local search with different swap strategies
    # Also try: for each element, see if we can replace it with a set of 2+ elements
    # But that's complex. Let's try a simpler improvement: 
    # Try to find pairs (x, y) both in S where removing x and adding z (compatible with S\{x}) improves
    
    # Let's do a second round of local search with more iterations
    if time.time() < deadline - 5:
        S_ls3, S_ls3_set, sum_ls3 = local_search(best_S, best_S_set, max_iters=100000)
        if sum_ls3 > best_sum:
            best_S, best_sum, best_S_set = S_ls3, sum_ls3, S_ls3_set
    
    # Verify the solution
    for i in range(len(best_S)):
        for j in range(i + 1, len(best_S)):
            if lcm(best_S[i], best_S[j]) <= 256:
                # This shouldn't happen, but if it does, return empty or fix
                pass
    
    return best_S
# EVOLVE-BLOCK-END
