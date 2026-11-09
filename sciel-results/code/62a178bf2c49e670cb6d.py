"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    result = []
    
    # For each prime, select optimal powers
    # p=2: exponents {1, 2, 4} -> {2, 4, 16}
    # p=3: exponents {1, 2} -> {3, 9}
    # p=5: exponents {1, 2} -> {5, 25}
    # others: {1} -> {p}
    
    selections = {
        2: [1, 2, 4],
        3: [1, 2],
        5: [1, 2],
    }
    
    for p in primes:
        if p in selections:
            for e in selections[p]:
                result.append(p ** e)
        else:
            result.append(p)
            
    # Verify distinctness of subset products? 
    # The construction guarantees it, but a quick check is safe if time permits.
    # However, 2^16 = 65536 products. Checking distinctness takes O(2^k).
    # 65536 is fast.
    
    if time.time() < deadline:
        products = set()
        k = len(result)
        # Use bitmasks to generate subset products
        # This might be slow in Python if not careful, but 2^16 is small.
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= result[i]
            if prod in products:
                return [a for a in range(2, 41) if all(a%d for d in range(2, int(a**0.5)+1))] # Fallback to primes
            products.add(prod)
            
    return result

# EVOLVE-BLOCK-END
