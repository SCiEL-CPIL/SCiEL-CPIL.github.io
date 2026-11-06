"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all pairwise LCMs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Identify all bad triples: three distinct elements a<b<c where
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Build conflict graph: edge between two elements if they appear together
    # in some bad triple. We want a maximum independent set in this graph.
    # Use bitsets for efficiency.
    adj = [0] * (n + 1)  # bit i set in adj[x] means x conflicts with i
    for a, b, c in bad_triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Bitset of all elements
    all_bits = (1 << (n + 1)) - 1
    all_bits &= ~(1 << 0)  # exclude index 0
    
    # Degree of each vertex
    degree = [0] * (n + 1)
    for v in range(1, n + 1):
        degree[v] = bin(adj[v] & all_bits).count('1')
    
    # Greedy initial solution: sort by degree ascending, add if no conflict with selected
    order = sorted(range(1, n + 1), key=lambda x: (degree[x], x))
    selected_bits = 0
    selected = []
    for v in order:
        if (adj[v] & selected_bits) == 0:
            selected_bits |= (1 << v)
            selected.append(v)
    
    best_set = set(selected)
    best_size = len(selected)
    
    # Try to improve via local search / branch and bound
    # Use a simple local search: try swapping elements out to bring in others
    
    def validate(s):
        """Check if set s has no bad triple."""
        s_list = sorted(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                for k in range(j + 1, len(s_list)):
                    a, b, c = s_list[i], s_list[j], s_list[k]
                    if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                        return False
        return True
    
    # Local search: try to add elements that were excluded
    excluded = [v for v in range(1, n + 1) if v not in best_set]
    
    # Try adding each excluded element, removing conflicting ones if beneficial
    improved = True
    while improved:
        improved = False
        current_set = set(best_set)
        for v in excluded:
            if v in current_set:
                continue
            # Find which elements in current_set conflict with v
            conflicts = [u for u in current_set if (adj[v] >> u) & 1]
            # If removing conflicts allows us to gain more than we lose
            if len(conflicts) < 1:
                # Can add v directly
                current_set.add(v)
                improved = True
            elif len(conflicts) == 1:
                # Swap: remove one conflict, add v
                # Try if the result is valid
                new_set = current_set - set(conflicts)
                new_set.add(v)
                if validate(new_set) and len(new_set) >= len(current_set):
                    current_set = new_set
                    improved = True
            elif len(conflicts) == 2:
                # Try removing 2 conflicts, adding v (net -1, not helpful unless it enables more)
                pass
        
        # Also try removing elements that allow multiple additions
        for v in list(current_set):
            if time.time() > deadline:
                break
            # Try removing v
            test_set = current_set - {v}
            # Try adding excluded elements that conflict only with v
            gained = 0
            for u in excluded:
                if u in test_set:
                    continue
                # Check if u conflicts with anything in test_set
                if (adj[u] & 0) == 0:  # placeholder
                    pass
                conflicts_u = [x for x in test_set if (adj[u] >> x) & 1]
                if not conflicts_u:
                    test_set.add(u)
                    gained += 1
            if gained > 1:  # net gain
                current_set = test_set
                improved = True
        
        if current_set != best_set:
            best_set = current_set
            best_size = len(best_set)
            excluded = [v for v in range(1, n + 1) if v not in best_set]
    
    # Final verification
    if not validate(best_set):
        # Fallback: greedy from scratch with different orderings
        best_set = set()
        for start_order in [order, list(reversed(order)), sorted(range(1, n+1))]:
            test_set = set()
            for v in start_order:
                if all((adj[v] >> u) & 1 == 0 for u in test_set):
                    test_set.add(v)
            if len(test_set) > len(best_set):
                best_set = test_set
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
