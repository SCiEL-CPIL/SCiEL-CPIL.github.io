"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def is_valid_addition(x, selected):
        """Check if x can be added to selected without creating a bad triangle."""
        n = len(selected)
        for i in range(n):
            a = selected[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, n):
                b = selected[j]
                g_ab = gcd_table[a][b]
                if g_ab == g_ax:
                    g_bx = gcd_table[b][x]
                    if g_bx == g_ab:
                        return False
        return True
    
    def greedy_construct(order):
        """Greedily construct a valid set from the given order."""
        selected = []
        for x in order:
            if is_valid_addition(x, selected):
                selected.append(x)
        return selected
    
    best_set = []
    best_size = 0
    
    # Try multiple strategies with different initial orderings
    num_restarts = 40
    max_time_per_restart = 2.0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if r % 4 == 0:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif r % 4 == 1:
            # Primes first, then composites
            primes = [x for x in range(2, N + 1) if all(x % p != 0 for p in range(2, int(x**0.5) + 1))]
            composites = [x for x in range(1, N + 1) if x not in primes]
            order = primes + composites
            random.shuffle(order)
        elif r % 4 == 2:
            # Powers of 2 first
            powers2 = [2**k for k in range(1, 8) if 2**k <= N]
            others = [x for x in range(1, N + 1) if x not in powers2]
            order = powers2 + others
            random.shuffle(order)
        else:
            # Sort by number of prime factors (fewer first)
            def omega(n):
                count = 0
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        count += 1
                        n //= d
                    d += 1
                if n > 1:
                    count += 1
                return count
            order = sorted(range(1, N + 1), key=omega)
            random.shuffle(order)
        
        current_set = greedy_construct(order)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
        
        # Local search phase
        improved = True
        local_deadline = min(start_time + max_time_per_restart * (r + 1), deadline)
        while improved and time.time() < local_deadline:
            improved = False
            
            # Try to add any element not in set
            for x in range(1, N + 1):
                if x in current_set:
                    continue
                if is_valid_addition(x, current_set):
                    current_set.append(x)
                    improved = True
                    break
            
            if not improved:
                # Try 1-1 swaps: remove one element, add another
                for _ in range(300):
                    if time.time() > deadline:
                        return best_set
                    x = random.randint(1, N)
                    if x in current_set:
                        continue
                    y = random.choice(current_set)
                    
                    new_set = [z for z in current_set if z != y]
                    if is_valid_addition(x, new_set):
                        current_set = new_set + [x]
                        improved = True
                        break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if x not in best_set and is_valid_addition(x, best_set):
            best_set.append(x)
    
    return best_set
# EVOLVE-BLOCK-END
