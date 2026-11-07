import time
import random

def construct():
    """
    Construct a simple undirected graph on vertices 0..14 maximizing edges
    subject to: no simple 3-edge path s-u-v-t has endpoints s,t of equal degree.
    """
    n = 15
    deadline = time.time() + 80.0  # Safety margin before 90s limit

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(all_edges)

    def get_neighbors(edge_set):
        neigh = [set() for _ in range(n)]
        for u, v in edge_set:
            neigh[u].add(v)
            neigh[v].add(u)
        return neigh

    def is_valid(edge_set):
        """Check if no simple 3-edge path s-u-v-t has deg(s) == deg(t)."""
        neigh = get_neighbors(edge_set)
        degs = [len(neigh[i]) for i in range(n)]
        
        for u, v in edge_set:
            # s must be a neighbor of u, not equal to v (since u-v is an edge)
            # t must be a neighbor of v, not equal to u (since u-v is an edge)
            # s != t for path simplicity
            for s in neigh[u]:
                if s == v:
                    continue
                for t in neigh[v]:
                    if t == u or t == s:
                        continue
                    if degs[s] == degs[t]:
                        return False
        return True

    # Start with K_{7,8} bipartite graph
    current_edges = set()
    for u in range(7):
        for v in range(7, 15):
            current_edges.add((u, v))

    # Verify initial state is valid
    if not is_valid(current_edges):
        # Fallback: empty graph is always valid
        current_edges = set()

    best_edges = set(current_edges)
    best_count = len(current_edges)

    # Greedy edge addition phase
    while time.time() < deadline:
        improved = False
        for e in all_edges:
            if e in current_edges:
                continue
            new_edges = set(current_edges)
            new_edges.add(e)
            if is_valid(new_edges):
                current_edges = new_edges
                if len(current_edges) > best_count:
                    best_edges = set(current_edges)
                    best_count = len(current_edges)
                improved = True
                break  # Restart to pick up new opportunities
        
        if not improved:
            # Try 1-for-1 swaps
            missing = [e for e in all_edges if e not in current_edges]
            if not missing:
                break
            
            # Try a limited number of random swaps
            max_swap_attempts = 2000
            for _ in range(max_swap_attempts):
                if time.time() > deadline:
                    break
                rem_e = random.choice(list(current_edges))
                add_e = random.choice(missing)
                
                new_edges = set(current_edges)
                new_edges.remove(rem_e)
                new_edges.add(add_e)
                
                if is_valid(new_edges):
                    current_edges = new_edges
                    if len(current_edges) > best_count:
                        best_edges = set(current_edges)
                        best_count = len(current_edges)
                    improved = True
                    break
            
            if not improved:
                # Try 2-for-2 swaps occasionally
                if len(current_edges) >= 2 and len(missing) >= 2:
                    max_2swap_attempts = 500
                    for _ in range(max_2swap_attempts):
                        if time.time() > deadline:
                            break
                        rem1, rem2 = random.sample(list(current_edges), 2)
                        add1, add2 = random.sample(missing, 2)
                        
                        new_edges = set(current_edges)
                        new_edges.discard(rem1)
                        new_edges.discard(rem2)
                        new_edges.add(add1)
                        new_edges.add(add2)
                        
                        if is_valid(new_edges):
                            current_edges = new_edges
                            if len(current_edges) > best_count:
                                best_edges = set(current_edges)
                                best_count = len(current_edges)
                            improved = True
                            break
                
                if not improved:
                    break  # No more improvements possible
    
    # Convert to list of [u, v] pairs with proper types
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
