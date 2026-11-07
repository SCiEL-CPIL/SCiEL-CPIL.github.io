import time
from itertools import combinations

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Prime factorization for each number 2-40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def factorize(n):
        """Return exponent vector as tuple for primes list."""
        exponents = []
        for p in primes:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            exponents.append(count)
        return tuple(exponents)
    
    # Precompute exponent vectors for all numbers 2-40
    exp_vectors = {}
    for n in range(2, 41):
        exp_vectors[n] = factorize(n)
    
    def products_distinct(candidate_set):
        """Check if all subset products of candidate_set are distinct."""
        products = {1}
        for n in candidate_set:
            new_products = {x * n for x in products}
            if new_products & products:
                return False
            products = products | new_products
        return True
    
    def can_add(current_set, new_num):
        """Check if adding new_num to current_set keeps all subset products distinct."""
        products = {1}
        for n in current_set:
            products = {x * n for x in products} | products
        new_products = {x * new_num for x in products}
        return len(new_products & products) == 0
    
    # Greedy approach with multiple orderings
    best_set = []
    
    # Try different sorting strategies
    strategies = [
        lambda x: x,  # ascending
        lambda x: -x,  # descending
        lambda x: (x in [2,4,8,16,32], x),  # powers of 2 first
        lambda x: (x in [3,9,27], x),  # powers of 3 first
        lambda x: sum(exp_vectors[x]),  # total exponent count
        lambda x: max(exp_vectors[x]),  # max exponent
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        ordered = sorted(range(2, 41), key=strategy)
        current_set = []
        for num in ordered:
            if time.time() > deadline:
                break
            if can_add(current_set, num):
                current_set.append(num)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
    
    # Local search: try to swap elements to improve
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_size = len(best_set)
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for j in range(2, 41):
                    if time.time() > deadline:
                        break
                    if j in best_set:
                        continue
                    # Try swapping best_set[i] with j
                    new_set = best_set[:]
                    new_set[i] = j
                    if products_distinct(new_set):
                        # Check if we can now add more
                        temp_set = new_set[:]
                        for k in range(2, 41):
                            if time.time() > deadline:
                                break
                            if k not in temp_set and can_add(temp_set, k):
                                temp_set.append(k)
                        if len(temp_set) > best_size:
                            best_set = temp_set
                            improved = True
                            break
                if improved:
                    break
    
    # Try randomized restarts with remaining time
    if time.time() < deadline - 5:
        import random
        for _ in range(50):
            if time.time() > deadline:
                break
            random_order = list(range(2, 41))
            random.shuffle(random_order)
            current_set = []
            for num in random_order:
                if can_add(current_set, num):
                    current_set.append(num)
            if len(current_set) > len(best_set):
                best_set = current_set[:]
    
    # Final verification
    if not products_distinct(best_set):
        # Fallback to known good solution
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    
    return [int(x) for x in best_set]
