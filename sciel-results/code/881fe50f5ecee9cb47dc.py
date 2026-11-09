"""Triple systems on 16 points without K(2,2,2)."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    # Precompute K(2,2,2) configurations
    # For each K(2,2,2), store its 8 triples
    # Also for each triple, store which K(2,2,2) it belongs to
    k222_list = []  # list of frozensets of 8 triples
    triple_to_k222 = {t: [] for t in all_triples}
    
    for six in combinations(range(n), 6):
        a, b, c, d, e, f = six
        partitions = [
            ((a,b),(c,d),(e,f)), ((a,b),(c,e),(d,f)), ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)), ((a,c),(b,e),(d,f)), ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)), ((a,d),(b,e),(c,f)), ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)), ((a,e),(b,d),(c,f)), ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)), ((a,f),(b,d),(c,e)), ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            x1, x2 = p1
            y1, y2 = p2
            z1, z2 = p3
            eight = set()
            for x in (x1, x2):
                for y in (y1, y2):
                    for z in (z1, z2):
                        eight.add(tuple(sorted((x, y, z))))
            eight_frozen = frozenset(eight)
            k222_list.append(eight_frozen)
            for t in eight:
                triple_to_k222[t].append(len(k222_list) - 1)
    
    best_result = None
    best_count = 0
    
    # Strategy: randomized greedy construction
    # Try multiple random orderings, greedily add triples if they don't create a K(2,2,2)
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Random ordering of triples
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        current = set()
        # Track for each K(2,2,2) how many of its 8 triples are in current
        k222_counts = [0] * len(k222_list)
        
        for t in shuffled:
            # Check if adding t would create any full K(2,2,2)
            would_create = False
            for k222_idx in triple_to_k222[t]:
                if k222_counts[k222_idx] == 7:
                    would_create = True
                    break
            
            if not would_create:
                current.add(t)
                for k222_idx in triple_to_k222[t]:
                    k222_counts[k222_idx] += 1
        
        if len(current) > best_count:
            best_count = len(current)
            best_result = [list(t) for t in current]
    
    if best_result is None:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
# EVOLVE-BLOCK-END
