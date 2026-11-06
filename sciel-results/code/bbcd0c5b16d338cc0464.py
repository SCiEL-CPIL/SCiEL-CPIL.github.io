import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin out of 60s
    
    N = math.factorial(12)
    
    # Precompute prime factorization of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^2 * 11
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 2, 1]
    
    # Generate all divisors of N
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    divisors.sort()
    
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    
    # We want to find a set of distinct divisors d1, ..., dk (k<=12)
    # such that product is N, and max(di) is minimized.
    # All di > 12.
    
    # Strategy: 
    # Iterate M (the max factor) in increasing order over valid_divs.
    # For each M, check if N can be decomposed into distinct factors from valid_divs <= M.
    # Use backtracking search in decreasing order of factors to prune effectively.
    
    def find_decomp_for_M(M):
        # Returns a list of factors if found, else None.
        # Factors must be distinct, > 12, <= M, product N.
        allowed = [d for d in valid_divs if d <= M]
        # Sort allowed in decreasing order for the search
        allowed.sort(reverse=True)
        
        n_allowed = len(allowed)
        
        # Precompute suffix products for pruning?
        # In decreasing order, if we are at index i, we can pick from allowed[i:].
        # The maximum product we can form using any subset of allowed[i:] is the product of all of them.
        # If that product < remaining, we fail.
        # Let's precompute suffix_prods where suffix_prods[i] = product(allowed[i:])
        suffix_prods = [1] * (n_allowed + 1)
        for i in range(n_allowed - 1, -1, -1):
            suffix_prods[i] = suffix_prods[i+1] * allowed[i]
            
        # We also need to ensure we don't pick more than 12 factors.
        # But since min factor > 12, 12 factors is the absolute max.
        # We can add a depth counter.
        
        def backtrack(remaining, idx, depth):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return []
            
            if idx >= n_allowed:
                return None
            
            if depth >= 12:
                return None
                
            # Prune: if the product of all remaining allowed factors is less than remaining
            if suffix_prods[idx] < remaining:
                return None
            
            # Option 1: Skip allowed[idx]
            res = backtrack(remaining, idx + 1, depth)
            if res is not None:
                return res
            
            # Option 2: Take allowed[idx]
            if remaining % allowed[idx] == 0:
                res = backtrack(remaining // allowed[idx], idx + 1, depth + 1)
                if res is not None:
                    return [allowed[idx]] + res
            
            return None
        
        res = backtrack(N, 0, 0)
        if res:
            return res
        return None
    
    # Iterate M over valid_divs in increasing order
    for M in valid_divs:
        if time.time() > deadline:
            break
        
        res = find_decomp_for_M(M)
        if res is not None:
            # Found a solution with max factor M.
            # Since we iterate M in increasing order, this is the minimum possible max factor.
            # The factors in res are in decreasing order (due to search), but order doesn't matter for return.
            return res
    
    # Fallback
    return [N]

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
    prod = 1
    for x in res:
        prod *= x
    print("Product:", prod)
    print("12!:", math.factorial(12))
    print("Max:", max(res))
    print("Distinct:", len(res) == len(set(res)))
    print("All > 12:", all(x > 12 for x in res))
