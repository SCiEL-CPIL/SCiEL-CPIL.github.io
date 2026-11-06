import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    limit = 512
    
    # Precompute lcm compatibility matrix would be 512x512 = 262144 entries, feasible
    # But let's be smart: for each number, precompute which numbers it's compatible with
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute compatibility: comp[i] = set of j where lcm(i,j) <= 512
    # Actually, let's just check on the fly but efficiently
    
    # Greedy: start with 1, then add numbers that are most "compatible"
    # A number x is compatible with set S if lcm(x, y) <= 512 for all y in S
    
    # Key insight: if we sort numbers by how restrictive they are,
    # we should add less restrictive ones first.
    # A number x is more restrictive if it has large prime factors.
    # Let's sort by: for each x, compute the number of y in [1,512] with lcm(x,y) <= 512
    # Add in order of decreasing compatibility count (most compatible first)
    
    # Compute compatibility score for each number
    scores = []
    for x in range(1, limit + 1):
        count = 0
        for y in range(1, limit + 1):
            if x * y // gcd(x, y) <= limit:
                count += 1
        scores.append((count, x))
    
    # Sort by decreasing compatibility count, then by value
    scores.sort(key=lambda t: (-t[0], t[1]))
    
    S = []
    S_set = set()
    
    for count, x in scores:
        if time.time() > deadline - 10:
            break
        # Check if x is compatible with all in S
        compatible = True
        for y in S:
            if x * y // gcd(x, y) > limit:
                compatible = False
                break
        if compatible:
            S.append(x)
            S_set.add(x)
    
    # Local search: try to swap elements to add more
    # Try removing one element and adding two others
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(S)):
            if time.time() > deadline - 5:
                break
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = S_set - {removed}
            
            # Try to add as many new elements as possible
            candidates = [x for x in range(1, limit + 1) if x not in S_temp_set]
            # Sort candidates by compatibility
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                compatible = True
                for y in S_temp:
                    if x * y // gcd(x, y) > limit:
                        compatible = False
                        break
                if compatible:
                    S_temp.append(x)
                    S_temp_set.add(x)
                    improved = True
            
            if len(S_temp) > len(S):
                S = S_temp
                S_set = S_temp_set
                improved = True
    
    return S
