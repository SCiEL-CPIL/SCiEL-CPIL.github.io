from collections import Counter

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 100 # 100s margin
    
    # Generate powerful numbers up to L
    # Start with L=10^7
    L = 10**7
    
    # Sieve square-free up to L^(1/3)
    max_b = int(L ** (1/3)) + 2
    is_sq_free = [True] * (max_b + 1)
    is_sq_free[0] = False
    is_sq_free[1] = True
    for i in range(2, int(max_b**0.5) + 1):
        if is_sq_free[i]:
            for j in range(i*i, max_b + 1, i*i):
                is_sq_free[j] = False
                
    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b**3
        max_a = int((L // b3) ** 0.5)
        for a in range(1, max_a + 1):
            powerful.add(a*a * b3)
            
    P = sorted(powerful)
    # Count sums
    counts = Counter()
    # Optimization: only a <= b
    # We can use two pointers or just nested loops
    # Nested loops:
    # for i, a in enumerate(P):
    #   for b in P[i:]:
    #     counts[a+b] += 1
     
    # This is O(M^2).
    # If M is too large, we might timeout.
    # Let's check M.
    if len(P) > 20000:
        # Might be too slow, but let's try with a smaller subset or just run and hope
        # Or reduce L
        pass
        
    # Run the counting
    # To speed up, we can use a list and update indices? No, sums are large.
    # Counter is okay.
    
    # We can add a time check inside the loop?
    # It's hard to break out of nested loops easily.
    # We can just run it. If L=10^7, it should be fast enough.
    
    for i in range(len(P)):
        a = P[i]
        # b must be >= a
        # We can iterate over P[i:]
        # This creates a slice, which is O(M). Bad.
        # Use index j from i to end.
        for j in range(i, len(P)):
            b = P[j]
            counts[a + b] += 1
            if time.time() > deadline:
                # This check is expensive.
                # Maybe check every 10000 iterations?
                pass
                
    # Find best N
    best_N, best_count = max(counts.items(), key=lambda x: x[1])
    
    # Reconstruct pairs
    pset = set(P)
    pairs = []
    for a in P:
        b = best_N - a
        if a <= b and b in pset:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
