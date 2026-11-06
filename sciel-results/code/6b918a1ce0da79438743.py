import time
from math import factorial
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    N = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    best_result = [N]
    
    # Generate all divisors of N that are > 12
    divisors = []
    for i in range(13, N + 1):
        if N % i == 0:
            divisors.append(i)
    divisors.sort()
    divisor_set = set(divisors)
    
    # For each M from small to large, check if a valid decomposition exists
    # with all factors in (12, M], distinct, count <= 12
    
    for M in range(13, 10000):
        if time.time() > deadline:
            break
        
        # Candidate factors: divisors of N in (12, M]
        candidates = [d for d in divisors if d <= M]
        
        if len(candidates) < 1:
            continue
        
        # Meet-in-the-middle:
        # Split candidates into two halves
        # Enumerate all subsets of first half, store by product
        # For each subset of second half, check if N/product exists in first-half products
        # and the combined set is valid
        
        n_cand = len(candidates)
        
        if n_cand <= 20:
            # Small enough for direct subset enumeration
            # Try all subsets of size 1 to 12
            found = False
            for size in range(1, min(13, n_cand + 1)):
                if time.time() > deadline:
                    break
                for subset in combinations(candidates, size):
                    prod = 1
                    for f in subset:
                        prod *= f
                        if prod > N:
                            break
                    if prod == N:
                        best_result = list(subset)
                        found = True
                        break
                    if prod > N:
                        break
                if found:
                    break
            if found:
                return best_result
        else:
            # Meet-in-the-middle for larger candidate sets
            mid = n_cand // 2
            first_half = candidates[:mid]
            second_half = candidates[mid:]
            
            # Enumerate all subsets of first_half, group by product
            # Only keep products that divide N
            first_products = {}  # product -> list of subsets (as tuples)
            
            n1 = len(first_half)
            for mask in range(1, (1 << n1)):
                if time.time() > deadline:
                    break
                prod = 1
                subset = []
                valid = True
                for i in range(n1):
                    if mask & (1 << i):
                        prod *= first_half[i]
                        if prod > N:
                            valid = False
                            break
                        subset.append(first_half[i])
                if not valid:
                    continue
                if N % prod != 0:
                    continue
                if prod not in first_products:
                    first_products[prod] = []
                first_products[prod].append(tuple(subset))
            
            # Now enumerate subsets of second_half
            n2 = len(second_half)
            found = False
            for mask in range(0, (1 << n2)):
                if time.time() > deadline:
                    break
                prod = 1
                subset = []
                valid = True
                for i in range(n2):
                    if mask & (1 << i):
                        prod *= second_half[i]
                        if prod > N:
                            valid = False
                            break
                        subset.append(second_half[i])
                if not valid:
                    continue
                if N % prod != 0:
                    continue
                
                needed = N // prod
                if needed in first_products:
                    for first_subset in first_products[needed]:
                        # Check distinctness: all elements in first_subset and subset must be distinct
                        # Since candidates are sorted and we split, and each element appears once,
                        # distinctness is automatic as long as no element appears in both halves
                        # (which it can't since first_half and second_half are disjoint)
                        combined = list(first_subset) + list(subset)
                        if 1 <= len(combined) <= 12:
                            # Verify product
                            check_prod = 1
                            for f in combined:
                                check_prod *= f
                            if check_prod == N and all(f > 12 for f in combined) and len(set(combined)) == len(combined):
                                best_result = combined
                                found = True
                                break
                if found:
                    break
            if found:
                return best_result
    
    return best_result
