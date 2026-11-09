import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = 55.0
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return set(divs)

    # Start with divisors of 360
    S = get_divisors(360)
    best_S = set(S)
    best_size = len(S)
    
    # Try to improve by swapping
    # If we remove x and add y, size stays same.
    # We want to increase size.
    # This is hard.
    
    # Alternative: Greedy construction from scratch
    S2 = set()
    # Order numbers by how many others they are compatible with?
    # Or just iterate 1..512
    # If we add 1, it's compatible with all.
    # If we add 2, compatible with all even and odd <= 256.
    
    # Let's try a greedy that prioritizes small numbers?
    # Small numbers have more multiples.
    S_greedy = set()
    for i in range(1, 513):
        valid = True
        for j in S_greedy:
            if lcm(i, j) > 512:
                valid = False
                break
        if valid:
            S_greedy.add(i)
            
    if len(S_greedy) > best_size:
        best_S = S_greedy
        best_size = len(S_greedy)

    # Try another greedy: descending?
    S_greedy2 = set()
    for i in range(512, 0, -1):
        valid = True
        for j in S_greedy2:
            if lcm(i, j) > 512:
                valid = False
                break
        if valid:
            S_greedy2.add(i)
            
    if len(S_greedy2) > best_size:
        best_S = S_greedy2
        best_size = len(S_greedy2)

    # Local search on best_S
    # Try to add any number
    added = True
    while added and time.time() < deadline:
        added = False
        for x in range(1, 513):
            if x in best_S:
                continue
            valid = True
            for y in best_S:
                if lcm(x, y) > 512:
                    valid = False
                    break
            if valid:
                best_S.add(x)
                best_size += 1
                added = True
                
    return list(best_S)
