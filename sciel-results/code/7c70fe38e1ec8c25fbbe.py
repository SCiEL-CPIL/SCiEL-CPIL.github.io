import random
import time

def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct."""
    k = len(nums)
    if k > 20:
        return False # Safety
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _build_product_set(nums):
    """Build the set of all 2^k subset products incrementally.
    Returns (product_set, valid) where valid is False if any collision found."""
    products = {1}
    for num in nums:
        new_products = set()
        for p in products:
            np = p * num
            if np in products or np in new_products:
                return products, False
            new_products.add(np)
        products.update(new_products)
    return products, True

def _can_add(products, num):
    """Check if adding num to a set with given product set keeps all products distinct."""
    new_products = set()
    for p in products:
        np = p * num
        if np in products or np in new_products:
            return None
        new_products.add(np)
    return new_products

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    
    # Known optimal candidate (size 16)
    best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not _verify_subset_products(best_set):
        # Fallback
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    best_size = len(best_set)
    best_result = list(best_set)
    
    # Pool of all valid numbers
    all_nums = list(range(2, 41))
    
    # 1. Try to extend the known best set by 1 using incremental check
    current_set = set(best_result)
    prod_set, valid = _build_product_set(best_result)
    if valid:
        for num in all_nums:
            if num not in current_set:
                new_prods = _can_add(prod_set, num)
                if new_prods is not None:
                    best_result.append(num)
                    best_size = len(best_result)
                    return best_result
    
    # 2. Fast randomized greedy search with incremental product sets
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s wall-clock limit
    
    random.seed(42)  # Deterministic for reproducibility
    
    # Strategy: Start with random sets of various sizes, try to grow greedily
    for iteration in range(100000):
        if time.time() - start_time > timeout:
            break
        
        # Random restart with varying initial size
        init_size = random.choice([12, 13, 14, 15, 16])
        current = set(random.sample(all_nums, min(init_size, len(all_nums))))
        
        # Build product set for current
        prod_set, valid = _build_product_set(list(current))
        if not valid:
            continue
        
        # Greedy growth: repeatedly try to add any element that fits
        changed = True
        while changed and time.time() - start_time < timeout:
            changed = False
            pool = [n for n in all_nums if n not in current]
            random.shuffle(pool)
            
            for num in pool:
                new_prods = _can_add(prod_set, num)
                if new_prods is not None:
                    current.add(num)
                    prod_set.update(new_prods)
                    changed = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_result = list(current)
                    break  # Restart pool check as set changed
        
        # If we found 17 or 18, we can stop early
        if best_size >= 18:
            break
        if best_size >= 17 and iteration > 200:
            # Give a few more iterations to potentially find 18
            if iteration > 500:
                break

    # 3. Local search: try swaps on the best found set to improve
    if best_size < 18:
        current = set(best_result)
        prod_set, valid = _build_product_set(list(current))
        if valid:
            for _ in range(50000):
                if time.time() - start_time > timeout:
                    break
                
                # Pick random element to remove
                to_remove = random.choice(list(current))
                current.discard(to_remove)
                
                # Rebuild product set without to_remove
                remaining = [x for x in current]
                prod_set, valid = _build_product_set(remaining)
                if not valid:
                    current.add(to_remove)
                    continue
                
                # Try to add a new element
                pool = [n for n in all_nums if n not in current]
                random.shuffle(pool)
                
                added = False
                for num in pool:
                    new_prods = _can_add(prod_set, num)
                    if new_prods is not None:
                        current.add(num)
                        prod_set.update(new_prods)
                        added = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_result = list(current)
                        break
                
                if not added:
                    current.add(to_remove)  # Restore

    # Ensure types
    result = []
    seen = set()
    for x in best_result:
        if isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 and x not in seen:
            seen.add(x)
            result.append(x)
            
    return result
