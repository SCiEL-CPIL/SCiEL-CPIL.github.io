import time
import random
from itertools import combinations

# Deadline guard
START_TIME = time.time()
DEADLINE = 280.0 # Safety margin

def _time_left():
    return DEADLINE - time.time()

def _is_clique_5(G, S):
    """Check if S (tuple of 5 vertices) is a clique in G (adjacency list of sets)."""
    v0, v1, v2, v3, v4 = S
    if v1 not in G[v0] or v2 not in G[v0] or v3 not in G[v0] or v4 not in G[v0]:
        return False
    if v2 not in G[v1] or v3 not in G[v1] or v4 not in G[v1]:
        return False
    if v3 not in G[v2] or v4 not in G[v2]:
        return False
    if v4 not in G[v3]:
        return False
    return True

def _is_independent_5(G, S):
    """Check if S is an independent set."""
    v0, v1, v2, v3, v4 = S
    if v1 in G[v0] or v2 in G[v0] or v3 in G[v0] or v4 in G[v0]:
        return False
    if v2 in G[v1] or v3 in G[v1] or v4 in G[v1]:
        return False
    if v3 in G[v2] or v4 in G[v2]:
        return False
    if v4 in G[v3]:
        return False
    return True

def _has_violation(G, N):
    """Exhaustive check for K5 or Independent 5-set. Returns (type, vertices) or (None, None)."""
    # Optimization: Iterate combinations
    # This is the bottleneck. For N=42, ~850k iterations.
    # We will use this for final verification.
    for S in combinations(range(N), 5):
        if _is_clique_5(G, S):
            return 'clique', S
        if _is_independent_5(G, S):
            return 'independent', S
    return None, None

def _repair_graph(G, N, max_steps=10000):
    """
    Local search to fix violations.
    G: list of sets
    N: number of vertices
    """
    steps = 0
    while steps < max_steps and _time_left() > 5.0:
        # Find a violation. 
        # Instead of exhaustive search every step (too slow), 
        # we can keep track of known violations or do a partial search.
        # For simplicity and robustness, let's do a random sampling of violations first.
        # If we find one, fix it. If we don't find one in many samples, do exhaustive.
        
        found = False
        # Try 100 random samples
        for _ in range(100):
            S = tuple(random.sample(range(N), 5))
            if _is_clique_5(G, S):
                # Fix: flip a random edge in S
                u, v = random.sample(S, 2)
                if v in G[u]:
                    G[u].remove(v)
                    G[v].remove(u)
                else:
                    G[u].add(v)
                    G[v].add(u)
                found = True
                break
            if _is_independent_5(G, S):
                u, v = random.sample(S, 2)
                if v in G[u]:
                    G[u].remove(v)
                    G[v].remove(u)
                else:
                    G[u].add(v)
                    G[v].add(u)
                found = True
                break
        
        if not found:
            # No violation found in random samples.
            # Perform exhaustive check to be sure.
            v_type, S = _has_violation(G, N)
            if v_type is None:
                return True
            else:
                # Found a violation via exhaustive search. Fix it.
                u, v = random.sample(S, 2)
                if v in G[u]:
                    G[u].remove(v)
                    G[v].remove(u)
                else:
                    G[u].add(v)
                    G[v].add(u)
        
        steps += 1
        if steps % 100 == 0 and _time_left() < 10.0:
            break

    # Final check
    v_type, _ = _has_violation(G, N)
    return v_type is None

def _paley_graph(q):
    residues = {(x * x) % q for x in range(1, q)}
    G = [set() for _ in range(q)]
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                G[i].add(j)
                G[j].add(i)
                edges.append((i, j))
    return q, edges, G

def construct():
    # Try to find a graph with N > 37
    # We know 37 works (Paley).
    # We will try 42, 41, 40, 39, 38.
    
    best_n, best_edges, best_G = 37, [], []
    # Initialize with Paley 37 as fallback
    n, edges, G = _paley_graph(37)
    best_n, best_edges, best_G = n, edges, G

    for N in [42, 41, 40, 39, 38]:
        if _time_left() < 30.0:
            break
            
        # Initialize graph for this N
        # Start with a random graph
        G = [set() for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    G[i].add(j)
                    G[j].add(i)
        
        # Run repair
        if _repair_graph(G, N, max_steps=5000):
            # Valid graph found!
            edges = []
            for i in range(N):
                for j in G[i]:
                    if i < j:
                        edges.append((i, j))
            best_n = N
            best_edges = edges
            best_G = G
            break # We found a larger one, no need to try smaller? 
                   # Actually, we want the LARGEST. So if 42 works, we stop.
                   # If 42 fails, we try 41.

    return best_n, best_edges
