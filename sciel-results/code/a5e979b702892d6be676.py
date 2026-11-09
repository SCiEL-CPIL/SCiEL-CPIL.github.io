"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_c5_fast(adj, n):
    """Check if graph has a C5 subgraph."""
    cycles = [
        (0,1,2,3,4), (0,1,2,4,3), (0,1,3,2,4), (0,1,3,4,2),
        (0,1,4,2,3), (0,1,4,3,2), (0,2,1,3,4), (0,2,1,4,3),
        (0,2,3,1,4), (0,2,4,1,3), (0,3,1,2,4), (0,3,4,2,1)
    ]
    adj_sets = [set(adj[i]) for i in range(n)]
    for combo in combinations(range(n), 5):
        for cyc in cycles:
            if (adj_sets[combo[cyc[0]]].issuperset({combo[cyc[1]], combo[cyc[4]]}) and
                adj_sets[combo[cyc[1]]].issuperset({combo[cyc[0]], combo[cyc[2]]}) and
                adj_sets[combo[cyc[2]]].issuperset({combo[cyc[1]], combo[cyc[3]]}) and
                adj_sets[combo[cyc[3]]].issuperset({combo[cyc[2]], combo[cyc[4]]}) and
                adj_sets[combo[cyc[4]]].issuperset({combo[cyc[3]], combo[cyc[0]]})):
                return True
    return False

def _has_independent_9(adj_sets, n):
    """Check if there is an independent set of size 9."""
    for combo in combinations(range(n), 9):
        ok = True
        for i in range(9):
            u = combo[i]
            for j in range(i+1, 9):
                if combo[j] in adj_sets[u]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _is_valid(n, edge_list):
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    if _has_c5_fast(adj, n):
        return False
    if _has_independent_9(adj, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0
    
    base_n = 32
    base_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    best_result = {'n': base_n, 'edges': base_edges}
    
    # Attempt to find n=33 via random search
    # This is a long shot but might find a non-trivial structure
    n_target = 33
    random.seed(42)
    
    for trial in range(100):
        if time.time() > deadline:
            break
            
        # Generate a random graph
        edges = set()
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.3:
                    edges.add((i, j))
                    
        # Check validity
        # This is expensive, so we only do it if we have time
        if time.time() + 5.0 > deadline:
            break
            
        if _is_valid(n_target, list(edges)):
            best_result = {'n': n_target, 'edges': list(edges)}
            break

    return best_result

# EVOLVE-BLOCK-END
