"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # A triple {a,b,c} is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Precompute for each pair (a,b) the set of c that would form a bad triple
    # Actually, for efficiency: for each pair (a,b) with lcm L, any c that is a divisor of L
    # and lcm(a,c)==L and lcm(b,c)==L would form a bad triple.
    
    # Better approach: for each element x, check if adding x creates a bad triple with any existing pair.
    # For x to form a bad triple with a,b in selected: lcm(a,b)==lcm(a,x)==lcm(b,x)
    # This means lcm(a,b) == lcm(a,x) == lcm(b,x) = L
    # So we need: lcm(a,b) == lcm(a,x) and lcm(a,b) == lcm(b,x)
    
    # Precompute: for each x and each a, what is lcm(a,x)? We have the table.
    # For a candidate x and existing selected set, check all pairs (a,b) in selected:
    #   if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x], reject x
    
    # Greedy with different orderings
    best_result = []
    
    # Try multiple orderings to find a large valid set
    orderings = [
        list(range(1, n + 1)),           # natural order
        list(range(n, 0, -1)),           # reverse natural
        list(range(1, n + 1, 2)),        # odds first
        list(range(2, n + 1, 2)),        # evens first
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),  # by popcount
    ]
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        selected = []
        selected_set = set()
        for x in order:
            if time.time() > deadline - 5:
                break
            valid = True
            # Check against all pairs in selected
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_result):
            best_result = selected
    
    # If we still have time, try a local search / improvement
    if best_result and time.time() < deadline - 10:
        current = set(best_result)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in range(1, n + 1):
                if x in current:
                    continue
                # Try adding x
                valid = True
                for a, b in combinations(current, 2):
                    if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                        valid = False
                        break
                if valid:
                    current.add(x)
                    improved = True
                    break
    
    return list(current) if current else best_result
# EVOLVE-BLOCK-END
