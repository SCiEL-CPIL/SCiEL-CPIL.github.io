import time
import random
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 85.0  # 5s margin before 90s kill
    
    N = 128
    # Precompute GCD table
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            G[i][j] = g
            G[j][i] = g
    
    # Precompute for each pair (i,j), the set of k such that adding k creates a bad triple
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(b,c) == gcd(a,c)
    # For a candidate x to be added to S, we need to check all pairs (a,b) in S:
    # if gcd(a,x) == gcd(b,x) and gcd(a,b) == gcd(a,x), then adding x creates a bad triple (a,b,x)
    
    # For speed, precompute for each x and each pair (a,b) whether (a,b,x) is a bad triple
    # bad_triple[a][b][x] = True if gcd(a,b)==gcd(a,x)==gcd(b,x)
    # This is 128^3 = 2M entries, manageable.
    
    # Actually, let's precompute: for each x, for each a, the set of b such that (a,b,x) is bad
    # Or better: for each x, precompute a "conflict" structure
    # For candidate x, it conflicts with pair (a,b) if gcd(a,x)==gcd(b,x)==gcd(a,b)
    
    # Let's precompute gcd matrix and use it directly.
    
    def is_valid_addition(S_set, S_list, x):
        """Check if x can be added to S without creating a bad triple."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            g_ax = G[a][x]
            for j in range(i + 1, n):
                b = S_list[j]
                g_bx = G[b][x]
                if g_ax == g_bx:
                    if G[a][b] == g_ax:
                        return False
        return True
    
    def full_validation(S_list):
        """Full validation of the entire set."""
        S_set = set(S_list)
        for x in S_list:
            S_list_without = [y for y in S_list if y != x]
            if not is_valid_addition(set(S_list_without), S_list_without, x):
                return False
        return True
    
    # Greedy initialization
    S_set = set()
    S_list = []
    for x in range(1, N + 1):
        if is_valid_addition(S_set, S_list, x):
            S_set.add(x)
            S_list.append(x)
    
    best_S_list = list(S_list)
    best_size = len(S_list)
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        current_size = len(S_list)
        
        # Try to add
        added = False
        for _ in range(100):
            x = random.randint(1, N)
            if x in S_set:
                continue
            if is_valid_addition(S_set, S_list, x):
                S_set.add(x)
                S_list.append(x)
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = list(S_list)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                added = True
                break
        
        if added:
            continue
        
        # Try to swap
        swapped = False
        for _ in range(100):
            idx_in = random.randint(0, len(S_list) - 1)
            x_in = S_list[idx_in]
            
            while True:
                x_out = random.randint(1, N)
                if x_out not in S_set:
                    break
            
            # Perform swap
            S_set.remove(x_in)
            S_list.pop(idx_in)
            
            if is_valid_addition(S_set, S_list, x_out):
                S_set.add(x_out)
                S_list.append(x_out)
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = list(S_list)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                swapped = True
                break
            else:
                # Restore
                S_set.add(x_in)
                S_list.append(x_in)
        
        if swapped:
            continue
        
        # Stuck: Shake up
        if S_list:
            num_remove = random.randint(1, 3)
            for _ in range(min(num_remove, len(S_list))):
                if S_list:
                    idx = random.randint(0, len(S_list) - 1)
                    x_to_remove = S_list[idx]
                    S_set.remove(x_to_remove)
                    S_list.pop(idx)
            no_improve_count += 10
        
        # If stuck for too long, restart from best
        if no_improve_count > 200:
            S_set = set(best_S_list)
            S_list = list(best_S_list)
            no_improve_count = 0
            # Perturb best solution
            if S_list:
                for _ in range(random.randint(1, 5)):
                    if S_list:
                        idx = random.randint(0, len(S_list) - 1)
                        x_to_remove = S_list[idx]
                        S_set.remove(x_to_remove)
                        S_list.pop(idx)
    
    # Final validation
    if not full_validation(best_S_list):
        # Fallback: use a known valid construction
        # Powers of 2: 1,2,4,8,16,32,64,128 - check if valid
        fallback = [1, 2, 4, 8, 16, 32, 64, 128]
        if full_validation(fallback):
            best_S_list = fallback
        else:
            # Just return single element
            best_S_list = [1]
    
    return list(best_S_list)
