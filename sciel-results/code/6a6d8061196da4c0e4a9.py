"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    
    # Get all primes up to n
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(n)
    
    # Initialize with 1 and the maximum prime power for each prime <= n
    # This ensures we start with a valid set of pairwise coprime integers
    selected = [1]
    selected_set = {1}
    
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        selected.append(power)
        selected_set.add(power)
    
    # Local search: try to merge subsets of the current set into a single composite number
    # if the composite is <= n, is coprime to the rest, and has a higher sum value.
    
    def is_coprime_to_all(x, current_set):
        for y in current_set:
            if x == y:
                continue
            if math.gcd(x, y) != 1:
                return False
        return True

    improved = True
    while improved:
        improved = False
        # Sort by value descending to prioritize larger merges? Or just iterate.
        # We check subsets of size 2 and 3. Size > 3 is rare because product grows fast.
        # Max product for size 3: 2*3*5=30. 2*3*7=42. 2*5*7=70. 3*5*7=105. 2*3*5*7=210.
        # Size 4: 2*3*5*7=210. 2*3*5*11=330 > 256.
        # So we only need to check subsets of size 2, 3, and maybe 4 (only 2,3,5,7).
        
        current_list = list(selected)
        
        # Check size 2
        for i in range(len(current_list)):
            if improved: break
            for j in range(i + 1, len(current_list)):
                a = current_list[i]
                b = current_list[j]
                if math.gcd(a, b) != 1:
                    continue
                prod = a * b
                if prod > n:
                    continue
                if prod <= a + b:
                    continue
                # Check if prod is coprime to all others
                others = [x for k, x in enumerate(current_list) if k != i and k != j]
                if is_coprime_to_all(prod, set(others)):
                    # Perform merge
                    selected = others + [prod]
                    selected_set = set(selected)
                    improved = True
                    break
        
        # Check size 3
        if not improved:
            for i in range(len(current_list)):
                if improved: break
                for j in range(i + 1, len(current_list)):
                    if improved: break
                    for k in range(j + 1, len(current_list)):
                        a = current_list[i]
                        b = current_list[j]
                        c = current_list[k]
                        # Quick gcd check
                        if math.gcd(a, b) != 1 or math.gcd(a, c) != 1 or math.gcd(b, c) != 1:
                            continue
                        prod = a * b * c
                        if prod > n:
                            continue
                        if prod <= a + b + c:
                            continue
                        others = [x for idx, x in enumerate(current_list) if idx != i and idx != j and idx != k]
                        if is_coprime_to_all(prod, set(others)):
                            selected = others + [prod]
                            selected_set = set(selected)
                            improved = True
                            break
        
        # Check size 4 (only possible for very small primes)
        if not improved:
            # Only check if we have 2,3,5,7 in the set
            if 2 in selected_set and 3 in selected_set and 5 in selected_set and 7 in selected_set:
                prod = 2 * 3 * 5 * 7
                if prod <= n:
                    others = [x for x in current_list if x not in [2, 3, 5, 7]]
                    if is_coprime_to_all(prod, set(others)):
                        if prod > 2 + 3 + 5 + 7:
                            selected = others + [prod]
                            selected_set = set(selected)
                            improved = True

    # Ensure we return Python ints in a list
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
