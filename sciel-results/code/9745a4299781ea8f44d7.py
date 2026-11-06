def construct():
    from math import gcd
    from itertools import combinations
    
    N = 512
    
    # Precompute lcm for all pairs
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Check if a set is valid
    def is_valid(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if lcm(S[i], S[j]) > N:
                    return False
        return True
    
    # Start with divisors of a highly composite number
    # Try multiple starting points
    candidates = []
    
    # Divisors of 360
    divs_360 = [d for d in range(1, N+1) if 360 % d == 0]
    candidates.append(divs_360)
    
    # Divisors of 480
    divs_480 = [d for d in range(1, N+1) if 480 % d == 0]
    candidates.append(divs_480)
    
    # Divisors of 420
    divs_420 = [d for d in range(1, N+1) if 420 % d == 0]
    candidates.append(divs_420)
    
    # Also try the seed
    candidates.append(list(range(1, 23)))
    
    best = []
    for cand in candidates:
        if is_valid(cand):
            if len(cand) > len(best):
                best = list(cand)
    
    # Try to greedily add elements to the best set
    changed = True
    while changed:
        changed = False
        current = set(best)
        for x in range(1, N+1):
            if x not in current:
                # Check if adding x keeps the set valid
                valid = True
                for y in current:
                    if lcm(x, y) > N:
                        valid = False
                        break
                if valid:
                    best.append(x)
                    current.add(x)
                    changed = True
                    break
    
    return best
