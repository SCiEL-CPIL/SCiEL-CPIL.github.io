import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = factorial(12)
    
    # Precompute all divisors of N that are > 12
    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    all_divs = get_divisors(N)
    valid_divs = [d for d in all_divs if d > 12]
    valid_divs.sort()
    
    best_result = [N] # Fallback
    
    # Iterate M over valid_divs
    # We want to minimize M, so we check in ascending order.
    for M in valid_divs:
        if time.time() > deadline:
            break
            
        # Candidates are valid divisors <= M
        # We can pre-filter or just filter in loop.
        # Since valid_divs is sorted, we can find the index.
        # But simple list comprehension is fine for ~100-200 divisors.
        candidates = [d for d in valid_divs if d <= M]
        
        # Sort descending to pick large factors first?
        # This helps reduce remaining product quickly.
        candidates.sort(reverse=True)
        
        # Precompute suffix products for pruning
        suffix_prod = [1] * (len(candidates) + 1)
        for i in range(len(candidates) - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            
        def dfs_opt(index, current_product, factors):
            if time.time() > deadline:
                return None
            
            if current_product == N:
                return factors[:]
            
            if index == len(candidates):
                return None
                
            if current_product > N:
                return None
                
            remaining_needed = N // current_product
            
            # Prune if even multiplying by all remaining candidates is not enough
            if suffix_prod[index] < remaining_needed:
                return None
                
            c = candidates[index]
            
            # Include c
            if remaining_needed % c == 0:
                factors.append(c)
                res = dfs_opt(index + 1, current_product * c, factors)
                if res is not None:
                    return res
                factors.pop()
                
            # Exclude c
            res = dfs_opt(index + 1, current_product, factors)
            if res is not None:
                return res
                
            return None

        res = dfs_opt(0, 1, [])
        if res is not None:
            # Verify solution just in case
            # Distinctness is guaranteed by index.
            # Product is guaranteed by base case.
            # Factors > 12 guaranteed by valid_divs.
            # Count <= 12 guaranteed by magnitude.
            return res
                
    return best_result
