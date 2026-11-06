import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s kill
    
    def elapsed():
        return time.time() - start_time

    N = 128
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Precompute conflict pairs
    # conflict_pairs[x] is a list of (a, b) with a < b such that 
    # gcd(x, a) == gcd(x, b) == gcd(a, b)
    # If x is added to a set S, and both a and b are in S, it's invalid.
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g_xa = gcd_table[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_table[x][b] == g_xa and gcd_table[a][b] == g_xa:
                    conflict_pairs[x].append((a, b))
    
    # Sort conflict pairs by frequency or just keep as is. 
    # For faster checking, we can convert to a set of encoded pairs or just list.
    # Since N is small (128), list iteration is fine.

    def can_add(x, in_set):
        """Check if x can be added to the set represented by in_set."""
        for a, b in conflict_pairs[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    def is_valid_set(in_set):
        """Verify the whole set is valid. Used for final check or debugging."""
        elems = [i for i in range(1, N + 1) if in_set[i]]
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, len(elems)):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        return False
        return True

    def greedy_fill(in_set, order):
        """Try to add elements from 'order' to in_set if valid."""
        added = 0
        for x in order:
            if elapsed() > DEADLINE:
                break
            if not in_set[x] and can_add(x, in_set):
                in_set[x] = True
                added += 1
        return added

    # 3. Initialization Strategies
    # We want a good initial set.
    # Strategy A: Primes and prime powers first (less likely to share GCDs > 1)
    # Strategy B: Random
    # Strategy C: Descending by largest prime factor
    
    def largest_prime_factor(n):
        lp = 1
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                lp = d
                temp //= d
            d += 1
        if temp > 1:
            lp = temp
        return lp

    orderings = [
        sorted(range(1, N + 1), key=lambda x: (-largest_prime_factor(x), x)),
        sorted(range(1, N + 1), key=lambda x: (x.bit_count(), x)), # Fewer factors first
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    
    random.seed(42)
    for _ in range(10):
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        orderings.append(rand_order)

    best_in_set = [False] * (N + 1)
    best_size = 0

    for order in orderings:
        if elapsed() > DEADLINE - 10:
            break
        
        in_set = [False] * (N + 1)
        greedy_fill(in_set, order)
        
        size = sum(in_set)
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
    # 4. Local Search / Refinement
    # We will use the best set found so far as the starting point for local search.
    # We will iterate:
    # 1. Try to add any remaining element.
    # 2. If stuck, try swaps (remove u, add v) to potentially unlock additions.
    # 3. If stuck for a while, perform a "kick" (remove random elements) and refill.
    
    in_set = best_in_set.copy()
    current_size = best_size
    
    # List of elements currently in set for fast iteration
    def get_active(in_set):
        return [i for i in range(1, N + 1) if in_set[i]]

    def get_inactive(in_set):
        return [i for i in range(1, N + 1) if not in_set[i]]

    no_improve_count = 0
    max_no_improve = 5000
    
    while elapsed() < DEADLINE - 5:
        active = get_active(in_set)
        inactive = get_inactive(in_set)
        
        if not inactive:
            break
            
        # Phase 1: Greedy Add
        added_any = False
        for x in inactive:
            if elapsed() > DEADLINE - 5:
                break
            if can_add(x, in_set):
                in_set[x] = True
                current_size += 1
                added_any = True
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
        
        if added_any:
            no_improve_count = 0
            continue
        
        # Phase 2: Swap Search
        # We are stuck. Try to swap an element u in active with v in inactive.
        # This might allow other elements to be added later.
        # To be efficient, we only accept swaps if they lead to a valid set.
        # We don't require the size to increase immediately, but we hope it unlocks adds.
        
        swap_found = False
        # Try random swaps
        for _ in range(100): # Limit swaps per phase to avoid wasting time
            if elapsed() > DEADLINE - 5:
                break
            if not active or not inactive:
                break
            u = random.choice(active)
            v = random.choice(inactive)
            
            # Check if S - {u} + {v} is valid
            # Temporarily remove u
            in_set[u] = False
            
            # Check if v can be added to the rest
            if can_add(v, in_set):
                in_set[v] = True
                swap_found = True
                # Update active/inactive lists? 
                # We'll just break out and let the next loop iteration refresh them.
                # Actually, we should check if this swap allows new additions immediately?
                # The next iteration of the while loop will do Phase 1.
                break
            else:
                # Revert
                in_set[u] = True
        
        if swap_found:
            no_improve_count = 0
            continue
            
        # Phase 3: Kick
        # If no swap worked, we are in a local maximum.
        # Remove k random elements to "loosen" the set, then refill.
        no_improve_count += 1
        if no_improve_count > max_no_improve or random.random() < 0.05:
            # Kick: remove 3-5 random elements
            k = random.randint(3, 5)
            to_remove = random.sample(active, min(k, len(active)))
            for u in to_remove:
                in_set[u] = False
                current_size -= 1
            
            no_improve_count = 0
            # The next loop iteration will try to add back, potentially different ones

    # Final Check and Return
    # Ensure the best set found is valid
    if is_valid_set(best_in_set):
        return [i for i in range(1, N + 1) if best_in_set[i]]
    
    # Fallback: Return a known valid set (e.g., all primes)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    return primes

