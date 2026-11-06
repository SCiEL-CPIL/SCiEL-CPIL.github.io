import time

def construct():
    start_time = time.time()
    time_limit = 80
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    candidates = list(range(2, 129))
    
    # 1. Prime Set
    primes = [x for x in candidates if is_prime(x)]
    
    # 2. Seed Greedy
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    seed_seq = values
    
    # Choose the longer base
    if len(primes) >= len(seed_seq):
        base_seq = primes
    else:
        base_seq = seed_seq
        
    # Compute products for base_seq
    def get_products(seq):
        prods = set()
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.add(p)
        return prods
        
    current_seq = base_seq[:]
    current_products = get_products(current_seq)
    
    # 3. Greedy Extension
    # Try to add any number not in current_seq
    # We must maintain strictly increasing order, so we just check if adding x 
    # to the sorted list causes a conflict.
    
    added = True
    while added and (time.time() - start_time < time_limit):
        added = False
        for x in candidates:
            if x in current_seq:
                continue
                
            # Check if x can be added
            # We need to simulate adding x to the sequence
            new_seq = sorted(current_seq + [x])
            
            # Check for conflicts in new_seq
            # We can do this by checking all subarray products
            # But we can be smarter: 
            # Only new subarrays involving x can conflict.
            # Or, just recompute all products if the list is small (< 40).
            # Max length is ~35. Subarrays ~ 600. 
            # Recomputing 600 products is fast.
            
            new_products = get_products(new_seq)
            
            # Check if all products are distinct
            # If len(new_products) == len(new_seq)*(len(new_seq)+1)//2, they are distinct.
            expected_count = len(new_seq) * (len(new_seq) + 1) // 2
            if len(new_products) == expected_count:
                # Valid addition
                current_seq = new_seq
                current_products = new_products
                added = True
                # Break to restart loop with new base? 
                # Or continue trying to add more?
                # If we add x, other numbers might become valid or invalid.
                # Breaking and restarting the while loop ensures we always add the "smallest" possible next?
                # No, we iterate x in increasing order.
                # So we add the smallest valid x first.
                break
                
    return current_seq

# EVOLVE-BLOCK-END
