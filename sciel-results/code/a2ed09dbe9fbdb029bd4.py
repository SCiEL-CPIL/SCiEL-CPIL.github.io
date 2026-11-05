import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    target_num = 5
    target_den = 31
    
    best_solution = None
    best_max = 10000001

    def update_best(sol):
        nonlocal best_solution, best_max
        if sol:
            mx = max(sol)
            if mx < best_max:
                best_max = mx
                best_solution = sol

    # Search for 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a is in [7, 18]
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
            
        # R = 5/31 - 1/a = (5a - 31) / (31a)
        N = 5 * a - 31
        D = 31 * a
        
        # If N <= 0, no solution (1/a >= 5/31)
        if N <= 0:
            continue
            
        # We need 1/b + 1/c = N/D
        # (Nb - D)(Nc - D) = D^2
        K = D * D
        
        # Find divisors x of K such that x < D
        # x = Nb - D
        # y = Nc - D
        # b = (x + D) / N
        # c = (y + D) / N
        
        # Iterate possible x. Since x < D, we iterate up to D-1.
        # But we only care about divisors.
        # Efficient way: iterate i from 1 to sqrt(K), if K % i == 0, check i and K//i.
        # We want x < D.
        
        found_for_a = False
        # To minimize c, we want to maximize x (since y = K/x, c = (y+D)/N).
        # So we should check larger x first.
        # But finding divisors in descending order is tricky.
        # We can just collect valid (b, c) and pick the one with min c.
        
        candidates = []
        
        # Iterate divisors
        # K can be large (up to 18^2 * 31^2 ~ 300^2 * 1000 ~ 10^7).
        # sqrt(K) ~ 3000.
        for i in range(1, int(K**0.5) + 1):
            if time.time() > deadline:
                break
            if K % i == 0:
                # i is a divisor
                # Check if i < D
                if i < D:
                    x = i
                    y = K // x
                    # Check if b is integer
                    if (x + D) % N == 0:
                        b = (x + D) // N
                        if b > a:
                            # Check if c is integer
                            if (y + D) % N == 0:
                                c = (y + D) // N
                                if c > b:
                                    candidates.append((c, [a, b, c]))
                
                # Check K//i
                j = K // i
                if j != i and j < D:
                    x = j
                    y = K // x
                    if (x + D) % N == 0:
                        b = (x + D) // N
                        if b > a:
                            if (y + D) % N == 0:
                                c = (y + D) // N
                                if c > b:
                                    candidates.append((c, [a, b, c]))
        
        if candidates:
            # Find min c
            min_c, sol = min(candidates, key=lambda x: x[0])
            update_best(sol)

    # If no 3-term solution found, search for 4-term?
    # Given time constraints and the likelihood that 3-term is optimal for this fraction,
    # we might skip 4-term if 3-term is found with small max.
    # But to be safe, if best_max is still large, try 4-term.
    
    if best_max > 500:
        # Search 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
        # This is much more expensive.
        # We can fix a, b and solve for c, d.
        # 1/c + 1/d = 5/31 - 1/a - 1/b
        # This is similar to the 3-term case.
        
        for a in range(7, 20):
            if time.time() > deadline:
                break
            for b in range(a + 1, 50):
                if time.time() > deadline:
                    break
                # R = 5/31 - 1/a - 1/b
                # R = (5ab - 31b - 31a) / (31ab)
                N = 5 * a * b - 31 * b - 31 * a
                D = 31 * a * b
                if N <= 0:
                    continue
                
                K = D * D
                candidates = []
                for i in range(1, int(K**0.5) + 1):
                    if time.time() > deadline:
                        break
                    if K % i == 0:
                        if i < D:
                            x = i
                            y = K // x
                            if (x + D) % N == 0:
                                c = (x + D) // N
                                if c > b:
                                    if (y + D) % N == 0:
                                        d = (y + D) // N
                                        if d > c:
                                            candidates.append((d, [a, b, c, d]))
                        j = K // i
                        if j != i and j < D:
                            x = j
                            y = K // x
                            if (x + D) % N == 0:
                                c = (x + D) // N
                                if c > b:
                                    if (y + D) % N == 0:
                                        d = (y + D) // N
                                        if d > c:
                                            candidates.append((d, [a, b, c, d]))
                
                if candidates:
                    min_d, sol = min(candidates, key=lambda x: x[0])
                    update_best(sol)

    if best_solution:
        return best_solution
    
    # Fallback
    return [8, 31, 248]
