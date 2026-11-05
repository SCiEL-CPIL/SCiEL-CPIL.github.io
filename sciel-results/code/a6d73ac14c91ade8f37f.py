"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = 128
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Identify all bad triples
    bad_triples = []
    vertex_triples = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                g_ab = gcd_table[a][b]
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    t = (a, b, c)
                    bad_triples.append(t)
                    vertex_triples[a].append(t)
                    vertex_triples[b].append(t)
                    vertex_triples[c].append(t)
                    
    # Heuristic: Iteratively remove vertices with highest conflict degree
    # We want to maximize the remaining set, so we remove the "most constrained" vertex.
    
    best_set = list(range(1, n + 1))
    best_size = len(best_set)
    
    # Try multiple random restarts with slight variations in tie-breaking
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        selected = set(range(1, n + 1))
        # Count how many bad triples are currently "active" (all 3 in selected)
        # Initially all are active.
        # We maintain a count of active triples per vertex.
        active_counts = {i: len(vertex_triples[i]) for i in range(1, n + 1)}
        active_triples = set(bad_triples)
        
        while active_triples:
            # Find vertex with max active count
            # To add randomness, pick from top 5 or so?
            # For determinism in this structural change, let's stick to max, 
            # but randomize ties.
            max_count = -1
            candidates = []
            for v in selected:
                if active_counts[v] > max_count:
                    max_count = active_counts[v]
                    candidates = [v]
                elif active_counts[v] == max_count:
                    candidates.append(v)
            
            if not candidates:
                break
                
            v = random.choice(candidates)
            selected.remove(v)
            
            # Update active triples
            # For each triple containing v, if it was active, it becomes inactive
            for t in vertex_triples[v]:
                if t in active_triples:
                    active_triples.remove(t)
                    # Update counts for other vertices in t
                    a, b, c = t
                    for u in (a, b, c):
                        if u != v:
                            active_counts[u] -= 1
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
            
    return best_set if best_set else [1]
# EVOLVE-BLOCK-END
