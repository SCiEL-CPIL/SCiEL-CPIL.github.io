"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 40, 'maximum_size': 18}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Generate primes from 2 to 40
    primes = [i for i in range(2, P['n'] + 1) if _is_prime(i)]
    
    # Find prime powers <= 40
    prime_powers = []
    for p in primes:
        power = p
        while power <= P['n']:
            prime_powers.append(power)
            power *= p
            
    # Start with all prime powers
    current_set = set(prime_powers)
    
    # Iteratively remove elements that cause collisions
    # Collision: a * b * ... = c where a, b, c are in the set
    # We check for a * b = c first (most common)
    # Then a * b * d = c
    
    changed = True
    while changed:
        changed = False
        elements = sorted(current_set)
        # Check for a * b = c
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                prod = elements[i] * elements[j]
                if prod in current_set and prod != elements[i] and prod != elements[j]:
                    # Remove the product (larger one)
                    current_set.remove(prod)
                    changed = True
                    break
            if changed:
                break
        
        # If no simple collision, check for a * b * c = d
        if not changed:
            elements = sorted(current_set)
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    for k in range(j + 1, len(elements)):
                        prod = elements[i] * elements[j] * elements[k]
                        if prod in current_set and prod > elements[k]:
                            current_set.remove(prod)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break

    # Verify distinct subset products
    selected = sorted(current_set)
    products = [1]
    for x in selected:
        new_products = [p * x for p in products]
        products.extend(new_products)
    
    if len(products) != len(set(products)):
        # Fallback to primes if verification fails
        return [int(p) for p in primes]
    
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
