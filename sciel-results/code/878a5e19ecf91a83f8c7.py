import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    # Build initial graph
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    def get_cliques_and_independents(adj, n):
        cliques = []
        independents = []
        # Find all K4s
        for combo in combinations(range(n), 4):
            is_clique = True
            for u, v in combinations(combo, 2):
                if v not in adj[u]:
                    is_clique = False
                    break
            if is_clique:
                cliques.append(combo)
                
        # Find all independent sets of size 4
        for combo in combinations(range(n), 4):
            is_independent = True
            for u, v in combinations(combo, 2):
                if v in adj[u]:
                    is_independent = False
                    break
            if is_independent:
                independents.append(combo)
                
        return cliques, independents

    # Precompute for initial graph
    cliques, independents = get_cliques_and_independents(adj, n)
    
    # Try to add vertices
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try to add a new vertex
        # Randomly select a subset of neighbors
        # We can bias towards balanced degree or just random
        # Let's try random neighbor set
        new_neighbors = set()
        for i in range(n):
            if random.random() < 0.5:
                new_neighbors.add(i)
        
        # Check validity
        valid = True
        # Check K4 in neighbors
        for c in cliques:
            if all(v in new_neighbors for v in c):
                valid = False
                break
        
        if valid:
            # Check independent set of size 4 in non-neighbors
            non_neighbors = set(range(n)) - new_neighbors
            for ind in independents:
                if all(v in non_neighbors for v in ind):
                    valid = False
                    break
        
        if valid:
            # Add vertex
            n += 1
            adj.append(set(new_neighbors))
            for i in new_neighbors:
                adj[i].add(n - 1)
            
            # Update cliques and independents
            # This is expensive, but we only do it if we successfully add a vertex
            # Which should be rare or require many attempts
            # Actually, updating incrementally is complex. 
            # Recalculating is O(n^4). For n=38, 38^4 is ~2M. 
            # If we add a vertex, we pay this cost.
            # If we fail many times, we don't pay.
            cliques, independents = get_cliques_and_independents(adj, n)
            
            # If we added a vertex, we can continue trying to add more
            # But the graph gets denser/sparser, making it harder.
            # Let's just keep going until time runs out or we stop adding.
            # To avoid getting stuck in a local minimum where no random config works,
            # we might need backtracking, but that's too complex.
            # Given the gap between 37 and 42, finding 38 is a good win.
            
    # Return the best graph found
    # We need to return (n, edges)
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges
