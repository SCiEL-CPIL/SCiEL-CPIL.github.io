import time
import random
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 50  # Safety margin for 90s limit
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute bad triples
    # bad_pairs_with[x] contains pairs (a, b) such that {a, b, x} is a bad triple
    # i.e., gcd(a,b) == gcd(b,x) == gcd(x,a)
    bad_pairs_with = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[b][c] == gab and gcd_table[c][a] == gab:
                    bad_pairs_with[c].append((a, b))
                    bad_pairs_with[a].append((b, c))
                    bad_pairs_with[b].append((a, c))
    
    # Convert to sets for faster lookup? 
    # Actually, checking if (a,b) is in list is O(L). L is small.
    # But we can use a set of frozensets or just tuples in a set.
    bad_pairs_set = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for p in bad_pairs_with[x]:
            bad_pairs_set[x].add(p)

    def can_add(S, x):
        """Check if x can be added to set S."""
        # x must not be in S
        if x in S:
            return False
        # Check all pairs (a,b) in bad_pairs_set[x]
        # If both a and b are in S, then adding x creates a bad triple
        for a, b in bad_pairs_set[x]:
            if a in S and b in S:
                return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def simulated_annealing(initial_set, time_limit):
        """
        Perform Simulated Annealing starting from initial_set.
        Returns the best set found.
        """
        current = set(initial_set)
        best = set(initial_set)
        best_size = len(current)
        
        start_time = time.time()
        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.9995
        
        # Precompute list of all elements for random selection
        all_elements = list(range(1, N + 1))
        
        while time.time() - start_time < time_limit:
            # Choose a random move
            # 60% chance to try adding, 40% chance to try removing
            if random.random() < 0.6:
                # Try to add a random element
                x = random.choice(all_elements)
                if x not in current and can_add(current, x):
                    current.add(x)
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                else:
                    # If cannot add, maybe try removing something else to enable it?
                    # For now, just skip or accept a removal
                    # Let's stick to simple moves: Add if possible, else Remove random
                    pass
            else:
                # Try to remove a random element
                if current:
                    x = random.choice(list(current))
                    current.remove(x)
                    # Acceptance logic for removal
                    # Energy change is -1 (size decreases)
                    # Delta E = new_size - old_size = -1
                    # Prob = exp(-1 / T)
                    if random.random() > (1.0 / (T + 1e-9)):
                        # Reject the move (revert)
                        current.add(x)
                    else:
                        # Accept the move
                        pass
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best

    # 1. Initial Greedy Solution
    # Use a few different orderings
    best_set = set()
    best_size = 0
    
    # Natural order
    s = greedy_build(range(1, N + 1))
    if len(s) > best_size:
        best_set = s
        best_size = len(s)
        
    # Reverse order
    s = greedy_build(range(N, 0, -1))
    if len(s) > best_size:
        best_set = s
        best_size = len(s)
        
    # Random orders
    for _ in range(5):
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_build(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

    # 2. Simulated Annealing
    # Run SA for a few iterations, restarting if needed
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Split time among a few SA runs
        num_sa_runs = 3
        time_per_run = (remaining_time - 2) / num_sa_runs
        
        for i in range(num_sa_runs):
            if time.time() > deadline - 2:
                break
            # Start from best found so far
            # Occasionally perturb the starting set
            if i > 0 and random.random() < 0.5:
                start_set = set(best_set)
                # Remove a few random elements to escape local optimum
                for _ in range(random.randint(1, 3)):
                    if start_set:
                        start_set.remove(random.choice(list(start_set)))
            else:
                start_set = set(best_set)
                
            sa_best = simulated_annealing(start_set, time_per_run)
            if len(sa_best) > best_size:
                best_set = sa_best
                best_size = len(sa_best)

    # 3. Final Greedy Polish
    # Try to add any remaining elements
    for x in range(1, N + 1):
        if x not in best_set and can_add(best_set, x):
            best_set.add(x)

    return list(best_set) if best_set else [1]
