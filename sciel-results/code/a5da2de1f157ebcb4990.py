import random

def construct():
    n = 256
    # 1. Build conflict graph
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    # 2. Initial Greedy Independent Set (Randomized to break ties)
    # We use a fixed seed for reproducibility in the function, 
    # but the logic is local search.
    
    def get_greedy_is():
        remaining = set(range(1, n + 1))
        result = set()
        # Randomize order for diversity if we ran multiple times, 
        # but here we just run once then local search.
        # To be safe, let's just use standard min-degree greedy.
        while remaining:
            min_deg_node = None
            min_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = v
            
            result.add(min_deg_node)
            to_remove = adj[min_deg_node] | {min_deg_node}
            remaining -= to_remove
        return result

    S = get_greedy_is()
    
    # 3. Local Search
    # Blocked[v] = number of neighbors of v in S
    Blocked = [0] * (n + 1)
    for u in S:
        for v in adj[u]:
            if v not in S:
                Blocked[v] += 1
    
    # Unblocked = {v not in S : Blocked[v] == 0}
    def get_unblocked():
        return {v for v in range(1, n + 1) if v not in S and Blocked[v] == 0}
    
    unblocked = get_unblocked()
    
    # Iterate until no improvement
    max_iterations = 1000 # Safety break
    for _ in range(max_iterations):
        if unblocked:
            # Add one node from unblocked
            # Pick the one with min degree in remaining? Or just any.
            # Adding any increases size.
            v = next(iter(unblocked))
            S.add(v)
            # Update Blocked for neighbors of v
            for w in adj[v]:
                if w not in S:
                    Blocked[w] += 1
            # Recalculate unblocked
            unblocked = get_unblocked()
        else:
            # Try kick-out
            improved = False
            # Iterate over a copy of S to allow modification
            for u in list(S):
                # Find candidates: neighbors of u not in S with Blocked[u] == 1
                # (They are blocked ONLY by u)
                candidates = []
                for v in adj[u]:
                    if v not in S and Blocked[v] == 1:
                        candidates.append(v)
                
                if len(candidates) >= 2:
                    # Remove u
                    S.remove(u)
                    # Add candidates
                    for v in candidates:
                        S.add(v)
                        for w in adj[v]:
                            if w not in S:
                                Blocked[w] += 1
                    improved = True
                    unblocked = get_unblocked()
                    break
            
            if not improved:
                break
                
    return list(S)
