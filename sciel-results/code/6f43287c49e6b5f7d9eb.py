import time
import random
import math
from itertools import combinations

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Moore bound calculation for regular graph of degree k and girth g
    # For bipartite graphs (which this construction produces), the Moore bound is:
    # n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^{(g/2 - 1)}) if g is even
    # n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^{(g-1)/2}) if g is odd
    # Actually, for general girth g, the bound is:
    # n >= 2 * sum_{i=0}^{floor((g-1)/2)} (k-1)^i
    # Let's compute this lower bound
    moore_bound = 2
    power = 1
    for i in range(1, (g // 2) + 1):
        power *= (k - 1)
        moore_bound += power
        if time.time() > deadline:
            break
    
    # The graph must be simple and k-regular.
    # The construction below is for bipartite graphs.
    # If g is odd, a bipartite graph cannot have a cycle of length g (odd).
    # The task requires "no cycle shorter than g".
    # If g is odd, the shortest possible cycle in a bipartite graph is even.
    # So if we construct a bipartite graph with girth >= g+1 (next even number), it satisfies girth >= g.
    # If g is even, we need girth >= g.
    
    target_girth = g
    if g % 2 == 1:
        target_girth = g + 1 # Bipartite graphs have even girth. If we ensure girth >= g+1, we satisfy >= g.
    
    # We will search for n starting from moore_bound
    # n must be even for the bipartite construction.
    n = moore_bound
    if n % 2 != 0:
        n += 1
    
    # In each step, n increases by 2 to maintain bipartite symmetry
    # But we can also just check all n >= moore_bound.
    # Let's iterate n from moore_bound upwards.
    
    while n < 50000: # Upper limit to prevent infinite loop
        if time.time() > deadline:
            break
            
        m = n // 2
        
        # We need to select k shifts s_1, ..., s_k from Z_m \ {0}
        # such that no short cycles exist.
        # A cycle of length 2r in this graph corresponds to a sequence of r shifts 
        # s_{i_1}, ..., s_{i_r} from the set S = {s_1, ..., s_k}
        # such that sum_{j=1}^r (-1)^{j-1} s_{i_j} == 0 mod m.
        # Specifically, for a 4-cycle: s_i - s_j + s_k - s_l = 0 => s_i + s_k = s_j + s_l.
        # For a 6-cycle: s_i - s_j + s_k - s_l + s_m - s_n = 0.
        
        # Greedy Randomized Search for shifts
        found_valid = False
        best_shifts = None
        
        # To speed up, we can try to pick shifts that are "large" or spread out.
        # Random search with verification.
        
        # We need to check cycles of length up to target_girth.
        # The number of walks of length L from a vertex is k^L.
        # For k=5, g=9, L=4 (for 8-cycles? No, girth 9 means no cycles < 9.
        # Bipartite girth is even. So we need to check cycles of length 4, 6, 8.
        # If target_girth is 9, we need to avoid 4, 6, 8 cycles.
        # If target_girth is 10, we need to avoid 4, 6, 8 cycles (and 10 is allowed? No, "shorter than g". So if g=10, we avoid 4,6,8. 10 is not shorter than 10.
        # Wait, "no cycle shorter than g". So if g=9, cycles of length 4,6,8 are forbidden. 9 is odd, so no 9-cycles in bipartite. 10 is allowed.
        # So we need to check even lengths < g.
        
        max_cycle_len_to_check = g - 1
        if max_cycle_len_to_check % 2 != 0:
            max_cycle_len_to_check -= 1
            
        # We verify by generating all walks of length up to max_cycle_len_to_check/2 
        # and checking if they close.
        # Actually, a simpler check for this specific algebraic structure:
        # We can check all combinations of shifts of length r = max_cycle_len_to_check / 2.
        # Number of combinations is k^r. For k=5, r=4, 5^4 = 625. This is fast.
        # For r=5, 5^5 = 3125. Still fast.
        
        r_max = max_cycle_len_to_check // 2
        
        # Generate candidate shifts
        # We want shifts to be distinct and non-zero.
        # Also, to avoid 4-cycles: s_i + s_k != s_j + s_l for distinct indices?
        # Not exactly. The cycle is u -> v -> u' -> v' -> u.
        # u_i -> v_{i+s_1} -> u_{i+s_1-s_2} -> v_{i+s_1-s_2+s_3} -> u_{i+s_1-s_2+s_3-s_4}.
        # We need i+s_1-s_2+s_3-s_4 = i => s_1 - s_2 + s_3 - s_4 = 0.
        
        # Let's use a randomized greedy approach.
        # Pick s_1 randomly.
        # Pick s_2 randomly, check if it creates a 4-cycle with s_1? No, 4-cycle needs 4 shifts.
        # A 4-cycle involves 4 edges. The shifts involved are s_a, s_b, s_c, s_d.
        # Condition: s_a - s_b + s_c - s_d = 0.
        
        # We will try multiple random restarts to find a valid set of shifts.
        
        valid_shifts = None
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Generate k random distinct non-zero shifts
            shifts = random.sample(range(1, m), k)
            
            # Check for short cycles
            is_valid = True
            # We need to check all sequences of length r from 2 to r_max
            # Note: r=1 corresponds to 2-cycles (multi-edges), which are avoided by distinct shifts.
            # r=2 corresponds to 4-cycles.
            # r=3 corresponds to 6-cycles.
            # etc.
            
            # Efficient check:
            # For each r, iterate through all k^r sequences of shifts.
            # Calculate sum = s_1 - s_2 + s_3 - ... + (-1)^{r-1} s_r.
            # If sum % m == 0, we have a cycle of length 2r.
            # We must ensure the sequence doesn't "backtrack" in a way that creates a shorter cycle?
            # Actually, if a walk of length 2r closes, it contains a cycle of length <= 2r.
            # If the walk is simple, the cycle is exactly 2r.
            # If the walk is not simple, it contains a shorter cycle.
            # So if we find ANY closing walk of length 2r <= max_cycle_len_to_check, it's invalid.
            
            # However, checking all k^r walks is exponential.
            # For k=5, r_max=4 (if g=9, check up to 8, so r=4), 5^4=625.
            # For k=5, r_max=5 (if g=11, check up to 10, so r=5), 5^5=3125.
            # This is very fast.
            
            for r in range(2, r_max + 1):
                # Generate all sequences of length r
                # Using itertools.product
                for seq in itertools.product(range(k), repeat=r):
                    s_sum = 0
                    for j, idx in enumerate(seq):
                        if j % 2 == 0:
                            s_sum += shifts[idx]
                        else:
                            s_sum -= shifts[idx]
                    if s_sum % m == 0:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                valid_shifts = shifts
                break
        
        if valid_shifts:
            # Construct the graph
            edges = []
            for u in range(m):
                for s in valid_shifts:
                    v = (u + s) % m
                    # u is in U (0..m-1), v is in V (m..2m-1)
                    edges.append((u, m + v))
            
            return n, edges
        
        n += 2 # Try next even number

    # If we fail to find a small one, fall back to the LUW construction
    # to ensure we return something valid, even if large.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# Import itertools for the new logic
import itertools
