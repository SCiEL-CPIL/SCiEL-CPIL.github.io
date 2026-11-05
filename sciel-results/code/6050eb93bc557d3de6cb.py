import time
import random
from math import gcd
from itertools import combinations

# Precompute GCD table
N = 128
gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        gcd_table[i][j] = g
        gcd_table[j][i] = g

def get_conflicts(S, x):
    """
    Returns the set of elements in S that form a bad triple with x.
    A triple (a, b, x) is bad if gcd(a,b) == gcd(a,x) == gcd(b,x).
    """
    conf = set()
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            g_ab = gcd_table[a][b]
            if gcd_table[a][x] == g_ab and gcd_table[b][x] == g_ab:
                conf.add(a)
                conf.add(b)
    return conf

def greedy_construct(order):
    S = set()
    for x in order:
        if not get_conflicts(S, x):
            S.add(x)
    return S

def local_search(S, deadline):
    S = set(S)
    best_S = set(S)
    
    # Precompute degrees for initial ordering if needed, but we pass S
    # We will iterate through potential additions and swaps
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any node with 0 conflicts
        added = False
        for x in range(1, N + 1):
            if x not in S:
                if not get_conflicts(S, x):
                    S.add(x)
                    added = True
                    break
        if added:
            improved = True
            if len(S) > len(best_S):
                best_S = set(S)
            continue # Restart loop to check other additions
            
        # 2. Try swaps
        # We want to swap y in S with x not in S if conflicts(S, x) == {y}
        # To avoid getting stuck in 2-cycles, we can use a random selection
        # or a heuristic.
        
        # Heuristic: Choose the swap that maximizes the number of "free" nodes
        # This is expensive, so we sample a few swaps.
        
        best_swap = None
        best_score = -1
        
        # Sample a subset of x not in S to speed up
        candidates = [x for x in range(1, N + 1) if x not in S]
        if not candidates:
            break
            
        # Shuffle to add randomness
        random.shuffle(candidates)
        
        for x in candidates:
            c = get_conflicts(S, x)
            if len(c) == 1:
                y = next(iter(c))
                # Simulate swap
                S_temp = set(S)
                S_temp.remove(y)
                S_temp.add(x)
                
                # Score: number of nodes z not in S_temp that can be added
                # This is O(N * |S|^2) which is too slow for every x.
                # Approximate score: 
                # How many nodes in S_temp have fewer conflicts?
                # Or just use a random choice among valid swaps.
                
                # Let's just pick the first valid swap found in a random order
                # This is a simple hill climbing with random restarts on swaps.
                best_swap = (x, y)
                break
                
        if best_swap:
            x, y = best_swap
            S.remove(y)
            S.add(x)
            improved = True
            # Size didn't change, so best_S size is same. 
            # But we might be in a better position.
            
        if not improved:
            # Stuck. Try a random perturbation: remove 2 random elements and retry
            if len(S) > 2:
                to_remove = random.sample(list(S), 2)
                for y in to_remove:
                    S.remove(y)
                # After removing, we can try to add back or new elements
                # The loop will try to add elements in the next iteration
            else:
                break
                
        # Safety: if we loop too many times without size increase, break?
        # The 'improved' flag is set for swaps too, so we might loop forever.
        # We need a counter for steps without size increase.
        
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Calculate degrees for greedy ordering
    degrees = [0] * (N + 1)
    # Count bad triples for each node
    # This is O(N^3) which is 128^3 ~ 2M, acceptable.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1
                    
    best_set = set()
    
    # Multiple restarts
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
            
        # Generate an order
        # Mix of degree-based and random
        if restart % 3 == 0:
            order = sorted(range(1, N + 1), key=lambda x: (degrees[x], x))
        elif restart % 3 == 1:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        else:
            # Noisy degree
            noisy = [(degrees[i] + rng.random(), i) for i in range(1, N + 1)]
            order = [x[1] for x in sorted(noisy)]
            
        S = greedy_construct(order)
        
        # Local search
        S = local_search(S, deadline)
        
        if len(S) > len(best_set):
            best_set = set(S)
            
    return list(best_set)
