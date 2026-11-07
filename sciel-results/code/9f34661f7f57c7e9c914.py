"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Construct a set of multiplicatively independent numbers
    # by selecting powers of distinct primes such that subset sums of exponents are distinct.
    
    selected = []
    
    # Prime 2: Powers 2, 4, 8, 16, 32. Exponents 1, 2, 3, 4, 5.
    # Select {2, 4, 16} -> Exponents {1, 2, 4}. Subset sums: 0,1,2,3,4,5,6,7. Distinct.
    selected.extend([2, 4, 16])
    
    # Prime 3: Powers 3, 9, 27. Exponents 1, 2, 3.
    # Select {3, 9} -> Exponents {1, 2}. Subset sums: 0,1,2,3. Distinct.
    selected.extend([3, 9])
    
    # Prime 5: Powers 5, 25. Exponents 1, 2.
    # Select {5, 25} -> Exponents {1, 2}. Subset sums: 0,1,2,3. Distinct.
    selected.extend([5, 25])
    
    # Other primes: 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # Only one power <= 40 for each.
    selected.extend([7, 11, 13, 17, 19, 23, 29, 31, 37])
    
    # Verify distinctness of subset products
    # (Optional but good for safety, though the construction guarantees it)
    k = len(selected)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= selected[i]
        if prod in products:
            # Should not happen
            break
        products.add(prod)
        
    return selected

# EVOLVE-BLOCK-END
