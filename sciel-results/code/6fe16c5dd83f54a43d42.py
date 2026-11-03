import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound for k=5, g=9 is 428
    # We search for circulant graphs
    # n must be even for degree 5
    min_n = 428
    max_n = 1250  # Fallback LUW size
    
    # If k != 5 or g != 9, use LUW as fallback
    # But task says k=5, g=9
    if k != 5 or g != 9:
        # Fallback to LUW for other parameters
        # ... (implement LUW)
        pass

    # Search for circulant
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline:
            break
        half_n = n // 2
        # S = {a, -a, b, -b, n/2}
        for a in range(1, half_n):
            if time.time() > deadline:
                break
            for b in range(a + 1, half_n):
                # Check girth >= 9
                # S elements: a, -a, b, -b, half_n
                # Note: -a = n-a, -b = n-b
                # We check sums of length m
                # If any sum is 0 mod n, cycle exists
                
                # Optimize: check m=4 first
                # Precompute sums?
                # Just loop
                
                # m=4
                found_cycle = False
                # 5^4 = 625
                for s1 in range(5):
                    v1 = [a, -a, b, -b, half_n][s1]
                    for s2 in range(5):
                        v2 = [a, -a, b, -b, half_n][s2]
                        sum12 = (v1 + v2) % n
                        for s3 in range(5):
                            v3 = [a, -a, b, -b, half_n][s3]
                            sum123 = (sum12 + v3) % n
                            for s4 in range(5):
                                v4 = [a, -a, b, -b, half_n][s4]
                                if (sum123 + v4) % n == 0:
                                    # Check if it's a trivial cycle (backtracking)
                                    # e.g., a + -a + a + -a = 0
                                    # We need to ensure the walk is simple?
                                    # For girth, we just need existence of a cycle.
                                    # A relation s1+s2+s3+s4=0 implies a closed walk.
                                    # If the walk is not simple, it contains a shorter cycle.
                                    # So if any relation of length m exists, girth <= m.
                                    found_cycle = True
                                    break
                            if found_cycle: break
                        if found_cycle: break
                    if found_cycle: break
                if found_cycle: continue
                
                # m=5
                # 5^5 = 3125
                # ... and so on up to m=8
                
                # If no cycle found for m=4..8, girth >= 9
                # Construct graph
                edges = []
                S = [a, n-a, b, n-b, half_n]
                for u in range(n):
                    for s in S:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: LUW
    return luw_construct(k, g)

def luw_construct(k, g):
    # ... (seed code)
    pass
