"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    from itertools import combinations
    
    N = 512
    
    # Start with all divisors of 480 (2^5 * 3 * 5), which gives 24 elements
    # and lcm of any two divides 480 <= 512
    base = 480
    divisors = []
    for i in range(1, N + 1):
        if base % i == 0:
            divisors.append(i)
    
    # Try to greedily add more numbers
    # A number x can be added if lcm(x, y) <= 512 for all y in current set
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    current = set(divisors)
    
    # Try adding each number from 1 to 512 not in current
    candidates = [x for x in range(1, N + 1) if x not in current]
    
    # Sort by how "compatible" they are (fewer conflicts)
    for x in candidates:
        can_add = True
        for y in current:
            if lcm(x, y) > N:
                can_add = False
                break
        if can_add:
            current.add(x)
    
    # Local search: try to swap out elements for better ones
    # Try removing one and adding two
    improved = True
    while improved:
        improved = False
        for to_remove in list(current):
            temp = current - {to_remove}
            # Try to add two numbers
            additions = []
            for x in range(1, N + 1):
                if x in temp:
                    continue
                can_add = True
                for y in temp:
                    if lcm(x, y) > N:
                        can_add = False
                        break
                if can_add:
                    additions.append(x)
            # Try pairs from additions
            for i in range(len(additions)):
                for j in range(i + 1, len(additions)):
                    a, b = additions[i], additions[j]
                    if lcm(a, b) > N:
                        continue
                    # Check a and b against temp
                    if all(lcm(a, y) <= N for y in temp) and all(lcm(b, y) <= N for y in temp):
                        new_set = temp | {a, b}
                        if len(new_set) > len(current):
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return list(current)
# EVOLVE-BLOCK-END
