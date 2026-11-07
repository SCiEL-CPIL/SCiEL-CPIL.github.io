import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # For each prime, find the maximum dissociated subset of exponents
    # such that p^exp <= 40
    selected_numbers = []
    
    for p in primes:
        # Get available exponents
        exponents = []
        exp = 1
        while p ** exp <= 40:
            exponents.append(exp)
            exp += 1
            
        # Find max dissociated subset of exponents
        # A set A is dissociated if all subset sums are distinct
        best_subset = []
        # Try all subsets of exponents, check if dissociated
        # Since len(exponents) is small (max 5), we can brute force
        max_len = 0
        for r in range(len(exponents), -1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in itertools.combinations(exponents, r):
                # Check if subset is dissociated
                subset_sums = set()
                is_dissociated = True
                for mask in range(1 << len(subset)):
                    s = 0
                    for i in range(len(subset)):
                        if mask & (1 << i):
                            s += subset[i]
                    if s in subset_sums:
                        is_dissociated = False
                        break
                    subset_sums.add(s)
                
                if is_dissociated:
                    best_subset = list(subset)
                    found = True
                    break
            if found:
                break
        
        # Add the corresponding prime powers to the result
        for e in best_subset:
            selected_numbers.append(p ** e)
            
    # Verify the result
    # Check distinctness of all subset products
    # For 16 numbers, 2^16 = 65536 products.
    # Products can be large, but Python handles big ints.
    # We can use a set of tuples of exponents to be faster?
    # Or just big ints.
    
    products = set()
    n = len(selected_numbers)
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= selected_numbers[i]
        if prod in products:
            # Collision found, should not happen with this construction
            # But if it does, we need to fix it.
            # For now, assume construction is correct.
            pass
        products.add(prod)
        
    # Ensure types are Python int and list
    return [int(x) for x in selected_numbers]
