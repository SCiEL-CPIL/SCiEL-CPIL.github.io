import time

def _is_valid_subset_products(nums):
    k = len(nums)
    if k == 0:
        return True
    products = set()
    # Use iterative doubling or bitmask
    # Bitmask is easy
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit
    
    # Initial known valid set of 16
    nums_2 = [8, 16, 32]
    nums_3 = [9, 27]
    nums_5 = [5, 25]
    other_primes = [7, 11, 13, 17, 19, 23, 29, 31, 37]
    current_set = set(nums_2 + nums_3 + nums_5 + other_primes)
    candidates = [n for n in range(2, 41) if n not in current_set]
    
    best_set = list(current_set)
    
    # Greedy addition
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates[:]:
            if time.time() > deadline:
                break
            new_set = current_set | {x}
            if _is_valid_subset_products(list(new_set)):
                current_set = new_set
                candidates.remove(x)
                improved = True
                best_set = list(current_set)
                
    # Local search (swap)
    # This might be expensive, so limit it
    swaps_made = 0
    while time.time() < deadline and swaps_made < 100:
        found_swap = False
        for x in list(current_set):
            if time.time() > deadline:
                break
            for y in candidates:
                if time.time() > deadline:
                    break
                new_set = (current_set - {x}) | {y}
                if _is_valid_subset_products(list(new_set)):
                    current_set = new_set
                    candidates.remove(y)
                    candidates.append(x)
                    swaps_made += 1
                    found_swap = True
                    best_set = list(current_set)
                    break
            if found_swap:
                break
        if not found_swap:
            break
            
    return [int(x) for x in best_set]
