from math import gcd

def construct():
    N = 256
    selected = []
    # Standard Greedy
    for x in range(N, 0, -1):
        is_coprime_with_all = True
        for y in selected:
            if gcd(x, y) != 1:
                is_coprime_with_all = False
                break
        if is_coprime_with_all:
            selected.append(x)
    
    # Local Search: Try to split composite numbers into prime powers
    # We iterate until no improvement is found
    improved = True
    while improved:
        improved = False
        new_selected = list(selected)
        for i in range(len(new_selected)):
            x = new_selected[i]
            if x == 1:
                continue
            
            # Factorize x
            factors = []
            temp = x
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    count = 0
                    while temp % d == 0:
                        temp //= d
                        count += 1
                    factors.append((d, count))
                d += 1
            if temp > 1:
                factors.append((temp, 1))
            
            if len(factors) <= 1:
                continue
                
            # Try to replace x with prime powers of its factors
            # For each prime p in factors, find largest p^k <= 256
            # Check if these new numbers are coprime to the rest of the set (excluding x)
            rest = [y for j, y in enumerate(new_selected) if j != i]
            
            replacements = []
            valid = True
            for p, _ in factors:
                # Find largest power of p <= 256
                max_power = 1
                p_power = 1
                while p_power * p <= 256:
                    p_power *= p
                    max_power = p_power
                    
                # Check if max_power is coprime to all in rest
                # And coprime to other replacements?
                # Since replacements are distinct primes, they are coprime to each other.
                is_ok = True
                for y in rest:
                    if gcd(max_power, y) != 1:
                        is_ok = False
                        break
                if not is_ok:
                    valid = False
                    break
                replacements.append(max_power)
            
            if valid:
                old_sum = x
                new_sum = sum(replacements)
                if new_sum > old_sum:
                    # Replace x with replacements
                    new_selected.pop(i)
                    new_selected.extend(replacements)
                    selected = new_selected
                    improved = True
                    break # Restart loop with new list
                    # Actually, better to update selected and break inner loop
                    # But we are modifying new_selected. 
                    # Let's just update selected and break the for loop to restart while
                    # Or simply update selected in place?
                    # Easier: update selected, set improved=True, break for loop.
        
        if improved:
            continue
            
    return selected
