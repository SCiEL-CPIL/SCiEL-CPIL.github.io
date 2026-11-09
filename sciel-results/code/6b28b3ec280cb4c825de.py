import math
from collections import defaultdict

def construct():
    # 1. Generate powerful numbers up to L
    L = 10**12
    # Powerful numbers are of form a^2 * b^3 with b square-free
    # We can just generate by iterating b (square-free) and a
    # Or iterate n and check? No, L is too big.
    # Iterate b up to L^(1/3) ~ 10^4
    # a up to sqrt(L / b^3)
    
    powerful = set()
    max_b = int(L ** (1/3)) + 2
    # Precompute square-free numbers?
    # Or just iterate b and check if square-free?
    # Simpler: Iterate all powerful numbers by construction.
    # n = a^2 * b^3.
    # To ensure uniqueness, we don't strictly need b square-free if we just add to a set.
    # But to be efficient, we can restrict b to square-free.
    # Actually, just generating a^2 * b^3 for all a,b and adding to set is fine.
    # Range of b: 1 to 10^4.
    # Range of a: 1 to sqrt(10^12 / b^3).
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful.add(n)
            else:
                break
                
    powerful_list = sorted(powerful)
    P = len(powerful_list)
    
    # 2. Select A_check
    K = 100
    A_check = powerful_list[:K]
    
    best_N = 0
    best_count = 0
    
    # 3. Search for best N
    # We iterate b in powerful_list
    # N = A_check[0] + b
    # Check count of a in A_check such that N - a in powerful
    
    # Optimization: Only check b such that N <= 2*L (always true)
    # And N - a >= 1
    
    for b in powerful_list:
        N = A_check[0] + b
        # N is at most 10^12 + 10^12 = 2*10^12
        # We only have powerful numbers up to L=10^12
        # So N - a must be <= L
        # N - a <= L => A_check[0] + b - a <= L
        # b <= L - A_check[0] + a
        # Since a is small, b must be <= L roughly.
        # If b > L - min(A_check), then N - a might be > L for some a?
        # No, we check `if N - a in powerful`.
        # `powerful` only contains numbers <= L.
        # So if N - a > L, it won't be in the set.
        
        count = 0
        for a in A_check:
            rem = N - a
            if rem > 0 and rem in powerful:
                count += 1
        
        if count > best_count:
            best_count = count
            best_N = N
            
    # 4. Find all pairs for best_N
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
