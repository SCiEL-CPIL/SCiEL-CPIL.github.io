import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Precompute conflict bitsets C[a][b]
    # C[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    C = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L = lcm_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == L and lcm_table[b][c] == L:
                    mask |= (1 << c)
            C[a][b] = mask
            C[b][a] = mask
            
    # Precompute for each vertex v, the set of vertices that conflict with v
    # conflict_with[v] = bitmask of u such that there exists w with (v,u,w) bad
    conflict_with = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            mask = C[a][b]
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                # (a,b,c) is bad, so a conflicts with b and c, etc.
                conflict_with[a] |= (1 << b) | (1 << c)
                conflict_with[b] |= (1 << a) | (1 << c)
                conflict_with[c] |= (1 << a) | (1 << b)
                mask ^= lsb
    
    def popcount(m):
        return bin(m).count('1')
        
    def is_valid_add(x, S):
        """Check if x can be added to set S (bitmask)."""
        temp_S = S
        while temp_S:
            lsb = temp_S & -temp_S
            a = lsb.bit_length() - 1
            if C[a][x] & S:
                return False
            temp_S ^= lsb
        return True

    def get_list(S):
        return [i for i in range(1, N + 1) if S & (1 << i)]

    # Fast greedy initialization
    def greedy_init(order):
        S = 0
        for x in order:
            if is_valid_add(x, S):
                S |= (1 << x)
        return S

    # Local search with "remove 1, add 2" moves (size +1)
    def local_search(S, deadline_ls):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline_ls:
                break
                
            # Try adding a single element
            for x in range(1, N + 1):
                if not (S & (1 << x)):
                    if is_valid_add(x, S):
                        S |= (1 << x)
                        improved = True
                        break
                        
            if improved:
                continue
                
            # Try "Remove 1, Add 2" - size increases by 1
            selected = get_list(S)
            found_improvement = False
            for a in selected:
                if time.time() > deadline_ls:
                    break
                new_S = S & ~(1 << a)
                # Try to add two elements x, y
                for x in range(1, N + 1):
                    if new_S & (1 << x):
                        continue
                    if not is_valid_add(x, new_S):
                        continue
                    temp_S = new_S | (1 << x)
                    for y in range(x + 1, N + 1):
                        if temp_S & (1 << y):
                            continue
                        if is_valid_add(y, temp_S):
                            S = temp_S | (1 << y)
                            improved = True
                            found_improvement = True
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
                    
        return S

    # Branch and bound solver for maximum independent set in conflict hypergraph
    # A set S is valid if no three elements form a bad triple
    # We model this as: for each pair (a,b) in S, C[a][b] & S must be empty
    # 
    # Upper bound: |S| + number of remaining vertices that can potentially be added
    # Stronger bound: use greedy coloring on the conflict graph to get an upper bound
    
    def branch_and_bound(available, current_set, current_size, best_size_ref, deadline_bb):
        """
        available: bitmask of vertices not yet decided (can be added or excluded)
        current_set: bitmask of vertices currently in the solution
        current_size: number of vertices in current_set
        best_size_ref: list containing [best_size] for mutable reference
        deadline_bb: time limit for this B&B call
        """
        if time.time() > deadline_bb:
            return
        
        if not available:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
            return
        
        # Upper bound: current_size + popcount(available)
        # If this doesn't exceed best, prune
        if current_size + popcount(available) <= best_size_ref[0]:
            return
        
        # Try to add a vertex from available
        # Choose vertex with highest degree in the remaining conflict graph for better pruning
        # For simplicity, pick the first available vertex
        lsb = available & -available
        v = lsb.bit_length() - 1
        
        # Option 1: Include v
        if is_valid_add(v, current_set):
            new_current = current_set | lsb
            new_available = available & ~lsb
            # Remove vertices that conflict with v from available
            # A vertex u can only be in the solution if no bad triple is formed
            # If we include v, we need to ensure that for any u in new_available,
            # adding u doesn't create a bad triple with two existing elements
            # This is complex, so we just check validity when adding
            branch_and_bound(new_available, new_current, current_size + 1, best_size_ref, deadline_bb)
        
        # Option 2: Exclude v
        branch_and_bound(available & ~lsb, current_set, current_size, best_size_ref, deadline_bb)

    # Simulated annealing with same-size moves to escape local optima
    def simulated_annealing(S, deadline_sa):
        current_S = S
        current_size = popcount(current_S)
        best_S = current_S
        best_size = current_size
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        
        while T > min_T and time.time() < deadline_sa:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try add
                candidates = [x for x in range(1, N + 1) if not (current_S & (1 << x))]
                if candidates:
                    x = random.choice(candidates)
                    if is_valid_add(x, current_S):
                        new_S = current_S | (1 << x)
                        delta = 1
                        if delta > 0 or random.random() < min(1.0, 1.0 / T):
                            current_S = new_S
                            if popcount(current_S) > best_size:
                                best_S = current_S
                                best_size = popcount(current_S)
            elif move_type < 0.6:
                # Try remove
                selected = get_list(current_S)
                if selected:
                    a = random.choice(selected)
                    new_S = current_S & ~(1 << a)
                    delta = -1
                    if delta > 0 or random.random() < min(1.0, 1.0 / T):
                        current_S = new_S
            else:
                # Try 1-swap (remove a, add x)
                selected = get_list(current_S)
                if selected:
                    a = random.choice(selected)
                    candidates = [x for x in range(1, N + 1) if not (current_S & (1 << x))]
                    if candidates:
                        x = random.choice(candidates)
                        new_S = current_S & ~(1 << a)
                        if is_valid_add(x, new_S):
                            new_S |= (1 << x)
                            delta = 0
                            if random.random() < min(1.0, 1.0 / T):
                                current_S = new_S
                                if popcount(current_S) > best_size:
                                    best_S = current_S
                                    best_size = popcount(current_S)
                
            T *= cooling
            
            # Periodic local search polish
            if random.random() < 0.05:
                polished = local_search(current_S, deadline_sa)
                if popcount(polished) > best_size:
                    best_S = polished
                    best_size = popcount(polished)
                    current_S = polished
                    
        return best_S, best_size

    # Initial Greedy with different orders
    best_S = 0
    best_size = 0
    
    # Order 1: Primes first
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
            
    primes = [x for x in numbers if is_prime(x)]
    others = [x for x in numbers if x not in primes]
    
    order1 = primes + others
    S1 = greedy_init(order1)
    S1 = local_search(S1, deadline - 5)
    if popcount(S1) > best_size:
        best_S = S1
        best_size = popcount(S1)
        
    # Order 2: Random restarts
    random.seed(42)
    for i in range(80):
        if time.time() > deadline - 15:
            break
        random.shuffle(numbers)
        S = greedy_init(numbers)
        S = local_search(S, deadline - 15)
        sz = popcount(S)
        if sz > best_size:
            best_S = S
            best_size = sz
            
    # Simulated annealing from best
    if time.time() < deadline - 3:
        sa_S, sa_size = simulated_annealing(best_S, deadline - 2)
        if sa_size > best_size:
            best_S = sa_S
            best_size = sa_size
        
        # Final local search polish
        best_S = local_search(best_S, deadline - 1)
        best_size = popcount(best_S)
        
    return get_list(best_S)
