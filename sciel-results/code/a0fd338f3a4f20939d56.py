# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def _check_6_subsets_fast(adj, n):
    """Fast check: does any 6-subset induce a regular subgraph?
    A 6-vertex regular induced subgraph has all degrees equal.
    Possible degrees: 0 (empty), 1, 2, 3, 4, 5 (complete).
    We check all C(n,6) subsets but with early termination."""
    for subset in combinations(range(n), 6):
        # Compute degrees within the 6-subset
        deg = [0] * 6
        for i in range(6):
            ui = subset[i]
            for j in range(6):
                if i != j and adj[ui][subset[j]]:
                    deg[i] += 1
        # Check if all degrees are equal
        d0 = deg[0]
        regular = True
        for d in deg[1:]:
            if d != d0:
                regular = False
                break
        if regular:
            return False
    return True

def _check_all_subsets(adj, n):
    """Full check: no subset of size >= 6 induces a regular subgraph."""
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            deg = [0] * k
            for i in range(k):
                ui = subset[i]
                for j in range(k):
                    if i != j and adj[ui][subset[j]]:
                        deg[i] += 1
            d0 = deg[0]
            regular = True
            for d in deg[1:]:
                if d != d0:
                    regular = False
                    break
            if regular:
                return False
    return True

def _try_add_vertex(adj, n, deadline):
    """Try to add a new vertex (index n) to a valid n-vertex graph.
    Returns (new_adj, success) where new_adj is (n+1)x(n+1)."""
    new_n = n + 1
    # Try different random connection patterns for the new vertex
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        new_adj = [[False] * new_n for _ in range(new_n)]
        # Copy old edges
        for i in range(n):
            for j in range(n):
                new_adj[i][j] = adj[i][j]
        
        # Randomly connect new vertex to old vertices
        for i in range(n):
            if random.random() < 0.5:
                new_adj[i][new_n - 1] = new_adj[new_n - 1][i] = True
        
        # Check if the new graph is valid (at least check 6-subsets)
        if _check_6_subsets_fast(new_adj, new_n):
            # Full check to be safe
            if _check_all_subsets(new_adj, new_n):
                return new_adj, True
    
    return None, False

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    
    # Start with a small valid graph
    # A path on 6 vertices is non-regular: degrees 1,2,2,2,2,1
    # Actually let's start with n=6 and a non-regular graph
    # Simplest: path graph P6
    n = 6
    adj = [[False] * n for _ in range(n)]
    # Path: 0-1-2-3-4-5
    for i in range(n - 1):
        adj[i][i + 1] = adj[i + 1][i] = True
    
    # Verify it's valid (it should be - only 6 vertices, and it's non-regular)
    # Degrees: 1,2,2,2,2,1 - not all equal, so valid
    
    # Now try to extend
    while time.time() < deadline - 15:
        new_adj, success = _try_add_vertex(adj, n, deadline)
        if success:
            adj = new_adj
            n += 1
        else:
            # Try a different strategy: restart from a random valid graph of size n
            found = False
            for restart in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Random graph on n vertices
                rand_adj = [[False] * n for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            rand_adj[i][j] = rand_adj[j][i] = True
                
                # Check if valid
                if _check_6_subsets_fast(rand_adj, n) and _check_all_subsets(rand_adj, n):
                    adj = rand_adj
                    found = True
                    break
            
            if not found:
                break
            # Try adding vertex again
            new_adj, success = _try_add_vertex(adj, n, deadline)
            if success:
                adj = new_adj
                n += 1
            else:
                break
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
