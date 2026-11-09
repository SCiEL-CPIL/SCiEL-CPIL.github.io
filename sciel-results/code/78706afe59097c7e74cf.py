import time
from math import factorial
from functools import lru_cache
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We need distinct integers > 12 whose product is target.
    # Minimize the largest factor.
    
    # Strategy:
    # 1. Generate all divisors of target that are > 12.
    # 2. Sort them.
    # 3. We want to find a subset of these divisors (distinct, all > 12)
    #    that multiplies to target, minimizing the maximum element in the subset.
    # 4. We can iterate the maximum allowed factor M from smallest to largest.
    #    For a given M, we check if there exists a subset of divisors <= M
    #    that multiplies to target.
    # 5. The first M for which such a subset exists gives the optimal answer.
    
    # Generate divisors
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    if target > 12:
        divisors.append(target)
    
    divisors = sorted(set(divisors))
    
    # Pre-compute a mapping from value to index for quick lookup? 
    # Not strictly necessary if we use the list directly.
    
    # We will perform a DFS search for each candidate M.
    # To make this efficient, we can use memoization on (remaining_product, start_index).
    # However, since M changes, the set of available divisors changes.
    # Instead, we can iterate M in increasing order.
    # For each M, we define the available set as divisors[i] where divisors[i] <= M.
    
    # Optimization:
    # If we find a valid set for M, that's the answer.
    # We can prune the DFS:
    # - If remaining product is 1, success.
    # - If start_index exceeds the limit for current M, fail.
    # - If the smallest available divisor > remaining product, fail (since divisors > 1).
    # - If the product of the remaining available divisors < remaining product, fail.
    
    # Let's implement a helper that checks if a subset of divisors[start:end]
    # can multiply to 'rem'.
    
    # To handle the changing 'end' for each M efficiently, we can just pass the limit.
    
    # We'll use a recursive function with lru_cache.
    # But the cache depends on 'end'. Since 'end' changes for every M,
    # we might need to clear the cache or include 'end' in the key.
    # Including 'end' in the key is fine.
    
    # Actually, iterating M one by one and clearing cache is expensive.
    # Alternative:
    # Since the number of divisors is not huge (~200-300?), we can try a different approach.
    # Let's count divisors first.
    
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Number of divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
    # Divisors > 12 will be slightly fewer.
    
    # A DFS over 792 items is too big for brute force subset product.
    # But we are constrained by product = target.
    # The branching factor is limited by divisors of 'rem'.
    
    # Let's stick to the DFS with pruning.
    # We iterate M from the smallest divisor > 12 up to the largest.
    
    # To speed up, we can pre-filter divisors that are "useful".
    # Any divisor d must be a factor of target.
    
    # Let's refine the DFS.
    # State: (rem, idx) where idx is the index in the global sorted divisors list.
    # Constraint: divisors[idx] must be <= current_M.
    
    # We can define a function can_form(rem, idx, max_idx)
    # where max_idx is the index of the largest divisor <= M.
    
    # Since we iterate M, max_idx increases.
    # We can cache results for (rem, idx).
    # If can_form(rem, idx) is True for a certain max_idx, it remains True for larger max_idx?
    # No, because we are looking for a subset of the first max_idx+1 elements.
    # Actually, if a solution exists using elements up to index k, it also exists using elements up to index k+1.
    # So if can_form(rem, idx, max_idx) is True, then can_form(rem, idx, max_idx+1) is True.
    # This monotonicity suggests we can potentially reuse computations.
    
    # However, the standard DFS with lru_cache on (rem, idx) assumes we can use ANY element from idx onwards.
    # If we restrict to <= M, we effectively truncate the list.
    
    # Let's just run the DFS for each M.
    # To avoid re-computing, we can observe that if M is small, the search space is small.
    # As M grows, the search space grows.
    
    # A better approach:
    # Instead of iterating M, we can perform a single DFS that tries to find a solution
    # minimizing the maximum element.
    # We can sort divisors.
    # We pick factors in increasing order? No, that doesn't minimize the max directly.
    
    # Let's stick to the iterative M approach but optimize the inner DFS.
    
    # Pre-compute divisors list.
    # For each M in divisors:
    #   Find max_idx such that divisors[max_idx] <= M.
    #   Check if subset of divisors[0:max_idx+1] multiplies to target.
    
    # The inner check:
    # def dfs(rem, start):
    #   if rem == 1: return True
    #   if start > max_idx: return False
    #   if divisors[start] > rem: return False  # since sorted, all subsequent are > rem
    #   # Pruning: product of all remaining divisors from start to max_idx must be >= rem
    #   # This is hard to compute quickly without pre-computation.
    
    # Let's try a different pruning:
    # If rem > 1, we must pick at least one factor.
    # The smallest factor we can pick is divisors[start].
    # If divisors[start] > rem, fail.
    
    # Also, we can use the fact that factors must be distinct.
    
    # Let's implement the DFS with lru_cache.
    # We need to clear the cache for each M?
    # If we use (rem, start) as key, and the limit max_idx is implicit in the "start" range?
    # No, the limit is a global constraint for the current M.
    # If we don't include max_idx in the key, the cache is invalid when M changes.
    # If we include max_idx in the key, the cache grows large.
    
    # Alternative:
    # Since we are minimizing the max factor, we can do a binary search on M?
    # Is the property "exists a subset with max <= M" monotonic?
    # Yes. If a solution exists with max <= M, it also exists with max <= M' for M' > M.
    # So we can binary search on the sorted list of divisors!
    
    # This is a huge improvement.
    # Low = 0, High = len(divisors) - 1.
    # We want the smallest index idx such that a solution exists using divisors[0:idx+1].
    
    # Wait, the "max factor" is the value, not the index.
    # Since divisors are sorted, minimizing the max value is equivalent to minimizing the max index used.
    
    # So, Binary Search on the index of the maximum allowed divisor.
    
    low = 0
    high = len(divisors) - 1
    best_idx = -1
    best_solution = None
    
    # We need a function that checks if a solution exists using divisors[0:limit_idx+1]
    # and returns the solution if found.
    
    # To make the check fast, we use DFS with memoization.
    # The memoization key is (rem, start_idx).
    # The limit is fixed for the duration of one check.
    # So we can define the DFS inside the check function, or pass limit as a constant.
    
    def check(limit_idx):
        # Check if we can form target using divisors[0:limit_idx+1]
        # Returns a list of factors if found, else None.
        
        # We need to ensure we find a valid set.
        # DFS with backtracking.
        # To speed up, we can use memoization for (rem, start_idx) -> bool (or solution).
        # Storing the solution in cache is memory heavy. Storing bool is better.
        # But we need the actual factors to return.
        # We can store the path or just return bool and reconstruct?
        # Reconstructing is tricky.
        # Let's store the next index to try or the factor chosen?
        # Or just do DFS without cache for the solution path, but use cache for feasibility?
        
        # Given the small number of divisors (~792) and the strong constraint (product = 12!),
        # the search space is very small.
        # The number of divisors of 12! is 792.
        # The number of divisors > 12 is slightly less.
        # The branching factor is the number of divisors of 'rem' that are in the allowed set.
        
        # Let's try a simple DFS first. If it's too slow, we add caching.
        
        # Optimization: Sort divisors.
        # Pruning:
        # 1. If rem == 1: success.
        # 2. If start > limit_idx: fail.
        # 3. If divisors[start] > rem: fail (since sorted).
        # 4. If the product of all remaining divisors (start to limit_idx) < rem: fail.
        #    Pre-compute suffix products?
        #    suffix_prod[i] = divisors[i] * divisors[i+1] * ... * divisors[limit_idx]
        #    This depends on limit_idx, so we can't pre-compute globally.
        #    But we can compute it on the fly or skip this pruning if it's complex.
        
        # Let's use a cache for (rem, start_idx) -> bool.
        # If True, we know a solution exists from this state.
        # We still need to find the actual factors.
        # We can do a second pass or store the choice.
        
        # Let's try storing the choice: choice[rem, start_idx] = next_factor or -1.
        # Actually, simpler: just do DFS and return the list.
        # With lru_cache on (rem, start_idx) returning the list, it might be okay.
        # But lists are not hashable for lru_cache return values? No, lru_cache caches return values, they don't need to be hashable, just the arguments.
        
        # However, returning a list from a cached function means we return the SAME list object.
        # If we modify it, we break the cache.
        # So we must return a tuple or copy.
        
        # Let's use a cache that stores the FIRST factor to pick, or just a bool.
        # If bool is True, we can reconstruct by re-running DFS without cache but guided by the fact that it's solvable?
        # No, that's wasteful.
        
        # Let's use a cache that stores the next index to use.
        # memo[rem, start_idx] = next_idx_to_try
        # If -1, no solution.
        
        memo = {}
        
        def dfs(rem, start_idx):
            if rem == 1:
                return []
            
            if start_idx > limit_idx:
                return None
            
            if (rem, start_idx) in memo:
                return memo[(rem, start_idx)]
            
            # Pruning
            if divisors[start_idx] > rem:
                memo[(rem, start_idx)] = None
                return None
            
            # Optional: Check if product of remaining < rem
            # We can approximate or skip for now.
            
            for i in range(start_idx, limit_idx + 1):
                d = divisors[i]
                if d > rem:
                    break
                if rem % d != 0:
                    continue
                
                # Try d
                sub_sol = dfs(rem // d, i + 1)
                if sub_sol is not None:
                    sol = [d] + sub_sol
                    memo[(rem, start_idx)] = sol
                    return sol
            
            memo[(rem, start_idx)] = None
            return None
        
        sol = dfs(target, 0)
        return sol
    
    # Binary search
    # We want the smallest limit_idx such that check(limit_idx) is not None.
    
    # Note: check() is expensive.
    # We can cache the results of check(limit_idx)?
    # Since we do binary search, we call check on different limit_idx.
    # The results are not easily reusable because the internal state (divisors considered) changes.
    # But we can just run binary search.
    # Number of steps: log2(792) ~ 10.
    # Each check runs a DFS.
    
    # Is the DFS fast enough?
    # In the worst case, it explores many branches.
    # But with pruning (d > rem, rem % d != 0), it should be fast.
    
    # Let's add a time guard.
    
    for _ in range(20):  # Binary search iterations
        if time.time() > deadline:
            break
        
        mid = (low + high) // 2
        if mid < 0:
            break
            
        sol = check(mid)
        
        if sol is not None:
            best_solution = sol
            best_idx = mid
            high = mid - 1
        else:
            low = mid + 1
        
        if low > high:
            break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: If binary search didn't find it (e.g. timeout or logic error),
    # try a direct search or return the single factor [target] if valid.
    # [target] is a valid solution (1 factor, > 12, product target).
    # Max factor = target.
    # This is the worst case, but valid.
    
    # Let's try to find a better one with a simple greedy or random if time permits?
    # Or just return [target].
    
    return [target]

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(res)
#     prod = 1
#     for x in res:
#         prod *= x
#     print(prod == factorial(12))
#     print(max(res))
