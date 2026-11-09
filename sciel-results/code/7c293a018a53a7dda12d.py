"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Precompute bad_triples_mask[a][b]
    # bad_triples_mask[a][b] is an integer bitset where bit c is 1 if {a,b,c} is a bad triple
    bad_triples_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    mask |= (1 << c)
            bad_triples_mask[a][b] = mask
            bad_triples_mask[b][a] = mask

    def greedy(ordering):
        selected = []
        selected_set = set()
        forbidden = 0  # Bitset of all c that would form a bad triple with some pair in selected
        
        for x in ordering:
            if time.time() > deadline:
                break
            if x in selected_set:
                continue
            # Check if x is forbidden
            if (forbidden >> x) & 1:
                continue
            
            # x is valid. Add it.
            selected.append(x)
            selected_set.add(x)
            
            # Update forbidden mask
            for a in selected[:-1]:
                forbidden |= bad_triples_mask[a][x]
        
        return selected

    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Generate orderings
    base_orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
    ]
    
    # Add random shuffles
    num_restarts = 2000
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i < len(base_orderings):
            ordering = base_orderings[i]
        else:
            ordering = list(range(1, n + 1))
            random.shuffle(ordering)
            
        selected = greedy(ordering)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected
    
    if not best_result:
        best_result = [1]
        
    return best_result
# EVOLVE-BLOCK-END
