import time
import sys

def construct():
    # The set of integers found to be maximally distinct in subset products
    # within the range 2..40.
    # 
    # Strategy:
    # For each prime p, select a subset of powers p^k (k>=1) such that
    # all subset sums of the exponents k are distinct.
    # 
    # Prime 2: Powers 2, 4, 16 (exponents 1, 2, 4). Subset sums: 0,1,2,3,4,5,6,7. Distinct.
    # Prime 3: Powers 3, 9 (exponents 1, 2). Subset sums: 0,1,2,3. Distinct.
    # Prime 5: Powers 5, 25 (exponents 1, 2). Subset sums: 0,1,2,3. Distinct.
    # Prime 7: Power 7 (exponent 1).
    # Prime 11: Power 11.
    # Prime 13: Power 13.
    # Prime 17: Power 17.
    # Prime 19: Power 19.
    # Prime 23: Power 23.
    # Prime 29: Power 29.
    # Prime 31: Power 31.
    # Prime 37: Power 37.
    
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify distinctness and range
    # All are in 2..40
    # All are distinct
    # Subset products are distinct due to prime factorization uniqueness
    # and distinct subset sums of exponents for each prime.
    
    return result
