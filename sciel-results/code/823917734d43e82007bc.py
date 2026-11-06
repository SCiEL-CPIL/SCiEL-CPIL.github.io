def construct():
    import time
    from math import gcd

    def lcm(a, b):
        return a * b // gcd(a, b)

    def is_valid_triple(a, b, c):
        # Check if lcm(a,b) == lcm(a,c) == lcm(b,c)
        l_ab = lcm(a, b)
        l_ac = lcm(a, c)
        if l_ab != l_ac:
            return True
        l_bc = lcm(b, c)
        return l_ab != l_bc

    # 1. Generate all prime powers in [1, 128]
    prime_powers = set()
    prime_powers.add(1)
    for p in range(2, 129):
        is_prime = True
        for i in range(2, int(p**0.5) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            val = p
            while val <= 128:
                prime_powers.add(val)
                val *= p
                
    selected = list(prime_powers)
    selected_set = set(selected)
    
    # 2. Greedily add remaining numbers
    remaining = [x for x in range(1, 129) if x not in prime_powers]
    
    # Sort remaining by number of prime factors? Smaller numbers might be easier to fit?
    # Or just iterate.
    
    changed = True
    while changed:
        changed = False
        for x in remaining:
            # Check if x can be added
            can_add = True
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i+1, len(selected)):
                    b = selected[j]
                    if not is_valid_triple(a, b, x):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                selected.append(x)
                selected_set.add(x)
                remaining.remove(x)
                changed = True
                
    return selected
