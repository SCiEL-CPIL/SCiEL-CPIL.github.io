import time
import random
from math import gcd

def construct():
    n = 128
    deadline = time.time() + 78.0  # Safety margin
    
    # 1. Precompute LCMs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_val = i * j // gcd(i, j)
            lcm_table[i][j] = lcm_val
            lcm_table[j][i] = lcm_val

    # 2. Identify Bad Triples
    # A triple (a, b, c) with a < b < c is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lcm_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lcm_ab and lcm_table[b][c] == lcm_ab:
                    bad_triples.append((a, b, c))

    # 3. Build Conflict Structure
    # conflicts[x] is a list of pairs (y, z) such that {x, y, z} is a bad triple
    # We store pairs as tuples (min, max) for consistency
    conflicts = {i: [] for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        # For a, the conflicting pair is (b, c)
        conflicts[a].append((b, c))
        # For b, the conflicting pair is (a, c)
        conflicts[b].append((a, c))
        # For c, the conflicting pair is (a, b)
        conflicts[c].append((a, b))

    # Convert lists to sets for faster lookup? 
    # Actually, we need to check if BOTH elements of the pair are in S.
    # A list of tuples is fine.
    
    # Optimization: Store conflicts as a list of lists of two ints for faster iteration
    conflict_lists = {i: [] for i in range(1, n + 1)}
    for i in range(1, n + 1):
        for pair in conflicts[i]:
            conflict_lists[i].append(pair)

    # 4. Helper Functions
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for y, z in conflict_lists[x]:
            if y in S and z in S:
                return False
        return True

    def greedy_fill(S, order=None):
        """Add elements from order to S if possible. Returns updated S."""
        if order is None:
            order = list(range(1, n + 1))
        added = True
        while added:
            added = False
            for x in order:
                if x not in S and can_add(x, S):
                    S.add(x)
                    added = True
        return S

    # 5. Local Search
    rng = random.Random(42)
    
    # Initial Solution: Greedy with different orders
    best_set = set()
    best_size = 0
    
    # Try a few initial strategies
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: len(conflict_lists[x])), # Least restrictive first
        sorted(range(1, n + 1), key=lambda x: -len(conflict_lists[x])), # Most restrictive first
    ]
    
    for order in orders:
        S = set()
        greedy_fill(S, order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Random initializations
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S = set()
        greedy_fill(S, order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Local Search: Kick and Refill
    current_set = best_set
    current_size = best_size
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 0.01
    
    # Precompute a static order for refill to be fast, or use random?
    # Using a static order based on degree might be better for consistency in refill
    refill_order = sorted(range(1, n + 1), key=lambda x: len(conflict_lists[x]))
    
    max_iterations = 100000
    iteration = 0
    
    while iteration < max_iterations and time.time() < deadline:
        iteration += 1
        
        # Kick: Remove 1 or 2 random elements
        if current_size < 2:
            break
            
        r = rng.random()
        if r < 0.7:
            # Remove 1
            elem_to_remove = rng.choice(list(current_set))
            current_set.remove(elem_to_remove)
        else:
            # Remove 2
            elems_to_remove = rng.sample(list(current_set), 2)
            current_set.remove(elems_to_remove[0])
            current_set.remove(elems_to_remove[1])
            
        # Refill
        # We try to add back elements. 
        # To be efficient, we can just call greedy_fill on the current set.
        # But greedy_fill iterates through ALL elements. 
        # A smarter refill: only consider elements not in S.
        # Let's implement a targeted refill.
        
        # Create a list of candidates: all elements not in current_set
        # We can just use the global refill_order and check membership
        # This is O(N * conflicts) per step. N=128 is small.
        
        greedy_fill(current_set, refill_order)
        
        new_size = len(current_set)
        delta = new_size - current_size
        
        if delta > 0:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_set = current_set.copy()
            temperature = max(min_temperature, temperature * cooling_rate)
        elif delta == 0:
            # Accept with some probability to explore
            if rng.random() < 0.1:
                current_size = new_size
            temperature = max(min_temperature, temperature * cooling_rate)
        else:
            # Accept worse with probability exp(delta / temperature)
            # delta is negative
            prob = 2.718281828 ** (delta / temperature)
            if rng.random() < prob:
                current_size = new_size
            temperature = max(min_temperature, temperature * cooling_rate)
            
    if not best_set:
        return [1]
        
    return sorted(best_set)
