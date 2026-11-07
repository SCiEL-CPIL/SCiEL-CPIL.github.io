"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The set of 16 integers with distinct subset products in range [2, 40]
    # Derived from prime powers with distinct subset exponent sums:
    # 2^1, 2^2, 2^4 -> 2, 4, 16
    # 3^1, 3^2     -> 3, 9
    # 5^1, 5^2     -> 5, 25
    # 7^1          -> 7
    # 11, 13, 17, 19, 23, 29, 31, 37
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify constraints to be safe
    if not all(isinstance(x, int) for x in result):
        return [2]
    if any(x < 2 or x > 40 for x in result):
        return [2]
    if len(set(result)) != len(result):
        return [2]
        
    # Verify distinct subset products
    # For k=16, 2^16 = 65536 products. This is fast.
    products = set()
    n = len(result)
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= result[i]
        if prod in products:
            # Fallback if verification fails (should not happen)
            return [2]
        products.add(prod)
        
    return result

# EVOLVE-BLOCK-END
