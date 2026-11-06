import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    FULL_MASK = (1 << N) - 1
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute conflicting_pairs[x]
    # conflicting_pairs[x] contains pairs (a, b) such that {x, a, b} is a bad triple.
    # We store a < b.
    conflicting_pairs = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                if gcd_table[b][c] == g_ab and gcd_table[c][a] == g_ab:
                    conflicting_pairs[a].append((b, c))
                    conflicting_pairs[b].append((a, c))
                    conflicting_pairs[c].append((a, b))

    def can_add(S_mask, x):
        """Check if x can be added to S_mask without creating a bad triple."""
        if S_mask & (1 << x):
            return False
        for a, b in conflicting_pairs[x]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return False
        return True

    def greedy_init():
        S = 0
        remaining = list(range(1, N+1))
        random.shuffle(remaining)
        for x in remaining:
            if can_add(S, x):
                S |= (1 << x)
        return S

    best_S = greedy_init()
    best_size = bin(best_S).count('1')
    
    current_S = best_S
    current_size = best_size
    
    steps = 0
    max_steps = 2000000
    no_improve_count = 0
    threshold = 5000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Randomly pick an element x to try to add
        x = random.randint(1, N)
        
        if not (current_S & (1 << x)):
            if can_add(current_S, x):
                current_S |= (1 << x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue
        
        # x is in current_S, or x is not in current_S but cannot be added.
        # Attempt a move involving removal to explore.
        
        if current_size == 0:
            current_S = greedy_init()
            current_size = bin(current_S).count('1')
            continue
            
        # Pick a random element y to remove
        # Extract active elements
        active = []
        m = current_S
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            active.append(idx)
            m ^= lsb
            
        if not active:
             continue

        y = random.choice(active)
        current_S &= ~(1 << y)
        current_size -= 1
        
        # Try to add x (if x is not y)
        if x != y:
            if can_add(current_S, x):
                current_S |= (1 << x)
                current_size += 1
                # Size change: -1 + 1 = 0.
                no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # We removed x (which was in S).
            no_improve_count += 1

        # If we haven't improved best_size in a while, restart from best_S
        if no_improve_count > threshold:
            current_S = best_S
            current_size = best_size
            no_improve_count = 0
            
        # Also, if current_size drops too low, restart
        if current_size < best_size - 5:
            current_S = best_S
            current_size = best_size
            no_improve_count = 0

    # Final Polish: Greedy Add
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for x in range(1, N+1):
            if not (best_S & (1 << x)):
                if can_add(best_S, x):
                    best_S |= (1 << x)
                    best_size += 1
                    improved = True

    result = [i for i in range(1, N+1) if best_S & (1 << i)]
    if not result:
        result = [1]
    return result
