import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute bad pairs for each x
    # bad_pairs[x] contains (a, b) such that (x, a, b) is a bad triple
    # We ensure a < b and a != x, b != x
    bad_pairs = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                g_bc = gcd_table[b][c]
                if g_ab == g_bc:
                    g_ca = gcd_table[c][a]
                    if g_ab == g_ca:
                        # Triple (a, b, c) is bad
                        # Add pairs to bad_pairs for each element
                        bad_pairs[a].append((b, c))
                        bad_pairs[b].append((a, c))
                        bad_pairs[c].append((a, b))

    def count_bad_triples(S):
        # S is a list of booleans or a set
        # This is O(N^3) if naive, but we use bad_pairs
        # Count each triple once. 
        # If we sum len of active pairs for all x, we count each triple 3 times.
        # So divide by 3.
        total = 0
        for x in range(1, N+1):
            if S[x]:
                for a, b in bad_pairs[x]:
                    if S[a] and S[b]:
                        total += 1
        return total // 3

    def get_delta_add(x, S):
        # Returns (delta_size, delta_bad_count)
        # delta_size = 1
        # delta_bad_count = number of pairs (a,b) in bad_pairs[x] where S[a] and S[b] are True
        cnt = 0
        for a, b in bad_pairs[x]:
            if S[a] and S[b]:
                cnt += 1
        return 1, cnt

    def get_delta_remove(x, S):
        # delta_size = -1
        # delta_bad_count = - (number of pairs (a,b) in bad_pairs[x] where S[a] and S[b] are True)
        # Note: Removing x breaks all bad triples containing x.
        # The number of such triples is the number of active pairs in bad_pairs[x].
        cnt = 0
        for a, b in bad_pairs[x]:
            if S[a] and S[b]:
                cnt += 1
        return -1, -cnt

    # Initialize S with a greedy result to start near a good solution
    # Greedy: Start with all, remove node with most bad triples
    S_init = [False] * (N+1)
    for i in range(1, N+1):
        S_init[i] = True
        
    # Quick greedy cleanup
    active = S_init[:]
    deg = [0] * (N+1)
    for x in range(1, N+1):
        for a, b in bad_pairs[x]:
            deg[x] += 1 # This counts each triple 3 times in total, but for greedy it's just a weight
    
    # Actually, let's just use the previous greedy_remove logic to get a good start
    # Or just start with all 1s and let SA fix it? 
    # Starting with all 1s has ~1000+ bad triples. SA might take too long to clear them all.
    # Let's run a simple greedy removal first.
    
    current_S = [False] * (N+1)
    for i in range(1, N+1):
        current_S[i] = True
        
    # Calculate initial bad triples count
    current_bad = count_bad_triples(current_S)
    
    # Greedy removal
    while current_bad > 0:
        max_deg = -1
        best_x = -1
        for x in range(1, N+1):
            if current_S[x]:
                # Count active bad triples containing x
                cnt = 0
                for a, b in bad_pairs[x]:
                    if current_S[a] and current_S[b]:
                        cnt += 1
                if cnt > max_deg:
                    max_deg = cnt
                    best_x = x
        if best_x == -1:
            break
        # Remove best_x
        current_S[best_x] = False
        # Recalculate bad count? 
        # We can update incrementally, but for simplicity in init, 
        # N=128 is small. 
        # But we want to be fast. 
        # Let's just recalculate if needed, or use the delta.
        # Delta remove:
        _, d_bad = get_delta_remove(best_x, current_S) # Note: S is still True for best_x here? 
        # My get_delta_remove assumes S[x] is True and we are removing it.
        # It counts pairs where OTHERS are active.
        # So current_bad += d_bad (which is negative)
        current_bad += d_bad
        # Update S
        # S[best_x] is already False.
        
    best_size = sum(current_S[1:])
    best_S = current_S[:]
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    steps_per_temp = 1000
    
    current_S_sa = current_S[:]
    current_bad_sa = current_bad
    current_size_sa = best_size
    
    while T > T_min and time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Pick random x to flip
            x = random.randint(1, N)
            
            if current_S_sa[x]:
                # Remove x
                d_size, d_bad = get_delta_remove(x, current_S_sa)
                d_energy = -d_size * 1000 + d_bad * 100
                # d_size is -1, so -d_size is 1. Energy increases by 1000? 
                # Wait. Energy = -Size * 1000 + Bad * 100.
                # If we remove x: Size decreases by 1 -> Energy increases by 1000.
                # Bad decreases by k -> Energy decreases by 100*k.
                # d_energy = 1000 - 100*k.
                
                # My formula: d_energy = -d_size * 1000 + d_bad * 100
                # d_size = -1. -(-1)*1000 = 1000. Correct.
                # d_bad = -k. (-k)*100 = -100k. Correct.
                
                if d_energy < 0 or random.random() < pow(2.718281828, -d_energy / T):
                    current_S_sa[x] = False
                    current_bad_sa += d_bad
                    current_size_sa += d_size
            else:
                # Add x
                d_size, d_bad = get_delta_add(x, current_S_sa)
                d_energy = -d_size * 1000 + d_bad * 100
                # d_size = 1. -1*1000 = -1000.
                # d_bad = k. k*100 = 100k.
                # d_energy = -1000 + 100k.
                
                if d_energy < 0 or random.random() < pow(2.718281828, -d_energy / T):
                    current_S_sa[x] = True
                    current_bad_sa += d_bad
                    current_size_sa += d_size
            
            # Track best valid solution
            if current_bad_sa == 0 and current_size_sa > best_size:
                best_size = current_size_sa
                best_S = current_S_sa[:]
        
        T *= cooling_rate
        
    return [i for i in range(1, N+1) if best_S[i]]
