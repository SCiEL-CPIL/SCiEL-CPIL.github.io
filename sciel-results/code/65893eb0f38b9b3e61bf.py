"""Improved program using structured local search with conflict tracking."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each element, store which other elements it has gcd=1 with
    # and group by gcd value for fast conflict checking
    
    def check_valid_addition(x, selected_set, selected_list):
        """Check if x can be added without creating equal-gcd triple."""
        for i in range(len(selected_list)):
            a = selected_list[i]
            g_ax = gcd_table[a][x]
            if g_ax == 1:
                continue
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                g_ab = gcd_table[a][b]
                if g_ab == g_ax:
                    if gcd_table[b][x] == g_ab:
                        return False
        return True
    
    def count_conflicts_if_added(x, selected_set, selected_list):
        """Count how many conflicts would be created if x is added."""
        count = 0
        for i in range(len(selected_list)):
            a = selected_list[i]
            g_ax = gcd_table[a][x]
            if g_ax == 1:
                continue
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                g_ab = gcd_table[a][b]
                if g_ab == g_ax:
                    if gcd_table[b][x] == g_ab:
                        count += 1
        return count
    
    def greedy_with_conflicts(order, max_conflicts_per_step=0):
        """Greedy construction that tracks conflicts."""
        selected = []
        selected_set = set()
        for x in order:
            if x in selected_set:
                continue
            if max_conflicts_per_step == 0:
                if check_valid_addition(x, selected_set, selected):
                    selected.append(x)
                    selected_set.add(x)
            else:
                # Allow some conflicts if we're in a repair phase
                conflicts = count_conflicts_if_added(x, selected_set, selected)
                if conflicts <= max_conflicts_per_step:
                    selected.append(x)
                    selected_set.add(x)
        return selected, selected_set
    
    # Multiple initialization strategies
    def init_primes_first():
        primes = [x for x in range(2, N + 1) if all(x % p != 0 for p in range(2, int(x**0.5) + 1))]
        composites = [x for x in range(1, N + 1) if x not in primes and x != 1]
        order = [1] + primes + composites
        random.shuffle(primes)
        random.shuffle(composites)
        order = [1] + primes + composites
        return order
    
    def init_by_prime_factor_count():
        # Sort by number of distinct prime factors (fewer first)
        def num_distinct_primes(n):
            count = 0
            d = 2
            while d * d <= n:
                if n % d == 0:
                    count += 1
                    while n % d == 0:
                        n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        order = sorted(range(1, N + 1), key=lambda x: (num_distinct_primes(x), x))
        random.shuffle(order)
        # Keep the ordering somewhat but add randomness
        result = []
        for i in range(0, N, 4):
            chunk = order[i:i+4]
            random.shuffle(chunk)
            result.extend(chunk)
        return result
    
    def init_random():
        order = elements[:]
        random.shuffle(order)
        return order
    
    def init_largest_first():
        order = elements[::-1]
        random.shuffle(order)
        return order
    
    # Local search: 2-opt and 3-opt moves
    def local_search(selected, selected_set, time_budget=2.0):
        """Perform local search to improve the set."""
        start = time.time()
        current_size = len(selected)
        best_selected = selected[:]
        best_set = selected_set.copy()
        best_size = current_size
        
        while time.time() - start < time_budget and time.time() < deadline:
            improved = False
            
            # Try 1-add moves
            candidates = [x for x in range(1, N + 1) if x not in selected_set]
            random.shuffle(candidates)
            for x in candidates:
                if check_valid_addition(x, selected_set, selected):
                    selected.append(x)
                    selected_set.add(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = selected[:]
                        best_set = selected_set.copy()
                    break
            
            if not improved:
                # Try 1-1 swaps
                if selected:
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        x_out = random.choice(selected)
                        x_in_candidates = [x for x in range(1, N + 1) if x != x_out and x not in selected_set]
                        if not x_in_candidates:
                            break
                        x_in = random.choice(x_in_candidates)
                        
                        # Check if swap is valid
                        new_selected = [z for z in selected if z != x_out]
                        new_set = selected_set - {x_out}
                        if check_valid_addition(x_in, new_set, new_selected):
                            selected = new_selected + [x_in]
                            selected_set = new_set | {x_in}
                            improved = True
                            break
            
            if not improved:
                # Try 2-2 swaps for more aggressive improvement
                if len(selected) >= 2:
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        x1_out, x2_out = random.sample(selected, 2)
                        x1_in_candidates = [x for x in range(1, N + 1) if x not in selected_set and x != x1_out and x != x2_out]
                        if len(x1_in_candidates) < 2:
                            break
                        x1_in, x2_in = random.sample(x1_in_candidates, 2)
                        
                        new_selected = [z for z in selected if z not in (x1_out, x2_out)]
                        new_set = selected_set - {x1_out, x2_out}
                        
                        if check_valid_addition(x1_in, new_set, new_selected):
                            new_selected.append(x1_in)
                            new_set.add(x1_in)
                            if check_valid_addition(x2_in, new_set, new_selected):
                                new_selected.append(x2_in)
                                new_set.add(x2_in)
                                selected = new_selected
                                selected_set = new_set
                                improved = True
                                break
                if not improved:
                    break
        
        return best_selected, best_set, best_size
    
    # Run multiple initialization strategies and local searches
    best_size = 0
    best_list = []
    best_set = set()
    
    init_strategies = [init_primes_first, init_by_prime_factor_count, init_random, init_largest_first]
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose initialization strategy
        strategy = init_strategies[r % len(init_strategies)]
        order = strategy()
        
        # Greedy construction
        selected, selected_set = greedy_with_conflicts(order, max_conflicts_per_step=0)
        current_size = len(selected)
        
        # Local search
        selected, selected_set, current_size = local_search(selected, selected_set, time_budget=1.5)
        
        if current_size > best_size:
            best_size = current_size
            best_list = selected[:]
            best_set = selected_set.copy()
    
    # Final polish: try to add any remaining
    for x in range(1, N + 1):
        if x not in best_set:
            if check_valid_addition(x, best_set, best_list):
                best_list.append(x)
                best_set.add(x)
                best_size += 1
    
    return best_list

# EVOLVE-BLOCK-END
