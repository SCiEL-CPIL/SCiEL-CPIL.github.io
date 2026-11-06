"""Rewritten program for bounded-LCM subset maximization."""
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    n = 512
    # Vertices 0..511 correspond to numbers 1..512
    
    # Precompute compatibility using bitsets
    # adj[i] is a bitset of vertices compatible with i (including i)
    adj = [0] * n
    
    for i in range(n):
        num_i = i + 1
        bits = 0
        for j in range(n):
            num_j = j + 1
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= n:
                bits |= (1 << j)
        adj[i] = bits
    
    # Greedy clique construction with multiple orderings
    best_clique = set()
    
    # Try multiple random-ish orderings
    import random
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        
        if trial == 0:
            order = list(range(n))
        elif trial == 1:
            order = list(range(n))
            order.reverse()
        else:
            order = list(range(n))
            rng.shuffle(order)
        
        # Greedy: pick vertices that are compatible with all already picked
        clique = set()
        # Start with a random vertex or first in order
        for v in order:
            if time.time() > deadline - 5.0:
                break
            # Check if v is compatible with all in clique
            compatible = True
            for u in clique:
                if not (adj[v] >> u) & 1:
                    compatible = False
                    break
            if compatible:
                clique.add(v)
        
        if len(clique) > len(best_clique):
            best_clique = clique.copy()
    
    # Local search: try to add vertices, swap to improve
    for iteration in range(5000):
        if time.time() > deadline - 3.0:
            break
        
        improved = False
        # Try adding any vertex not in clique
        for v in range(n):
            if v in best_clique:
                continue
            compatible = True
            for u in best_clique:
                if not (adj[v] >> u) & 1:
                    compatible = False
                    break
            if compatible:
                best_clique.add(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one vertex, add two
        for remove_v in list(best_clique):
            if time.time() > deadline - 3.0:
                break
            candidate_clique = best_clique - {remove_v}
            # Try to add two vertices
            added = 0
            for v in range(n):
                if v in candidate_clique or v == remove_v:
                    continue
                compatible = True
                for u in candidate_clique:
                    if not (adj[v] >> u) & 1:
                        compatible = False
                        break
                if compatible:
                    candidate_clique.add(v)
                    added += 1
                    if added == 2:
                        break
            if added == 2 and len(candidate_clique) > len(best_clique):
                best_clique = candidate_clique
                break
    
    # Convert to list of numbers
    result = sorted([v + 1 for v in best_clique])
    return result
