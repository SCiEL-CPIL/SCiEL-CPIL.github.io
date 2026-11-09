"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # A set of 16 integers in [2, 40] with distinct subset products.
    # This set consists of 8 large primes and 8 carefully selected small composites/primes
    # that are multiplicatively independent (1-independent).
    
    # Large primes:
    large_primes = [11, 13, 17, 19, 23, 29, 31, 37]
    
    # Small numbers:
    # 2, 4, 16 are powers of 2. 2*4=8 (not in set), 2*16=32 (not in set), 4*16=64.
    # 3, 27 are powers of 3. 3*27=81.
    # 5, 25 are powers of 5. 5*25=125.
    # 7 is prime.
    # Cross products:
    # 2*3=6 (not in set), 2*5=10, 2*7=14, 2*27=54, 2*25=50, 2*16=32.
    # 3*5=15, 3*7=21, 3*25=75, 3*16=48.
    # 4*3=12, 4*5=20, 4*7=28, 4*27=108, 4*25=100, 4*16=64.
    # 5*7=35, 5*27=135, 5*16=80.
    # 7*27=189, 7*25=175, 7*16=112.
    # 16*27=432, 16*25=400.
    # 25*27=675.
    # None of these products are in the set, nor are they equal to any single element in the set.
    # Also, no element is a product of a subset of the others.
    
    small_numbers = [2, 3, 4, 5, 7, 9, 16, 25]
    
    # Wait, I used 9 in the manual check above?
    # Let's re-verify the set: [2, 3, 4, 5, 7, 9, 16, 25]
    # 9 is 3^2.
    # Check 3 and 9: 3*9=27 (not in set).
    # Check 9 and 27? 27 not in set.
    # So 9 is safe.
    # Why did I switch to 27 in the thought process?
    # 3, 9, 27 is invalid because 3*9=27.
    # But 3, 9 is valid.
    # So [2, 3, 4, 5, 7, 9, 16, 25] is a valid set of 8 smalls.
    
    result = large_primes + small_numbers
    
    # Verify types
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
