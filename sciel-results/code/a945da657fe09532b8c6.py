import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial graph: K(7,8)
    # Vertices 0-6 in A, 7-14 in B
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    def get_degrees(edge_set):
        degs = [0] * n
        for u, v in edge_set:
            degs[u] += 1
            degs[v] += 1
        return degs
    
    def get_neighbors(edge_set):
        neighbors = [set() for _ in range(n)]
        for u, v in edge_set:
            neighbors[u].add(v)
            neighbors[v].add(u)
        return neighbors
    
    def is_valid(edge_set, degs, neighbors):
        # Check all paths s-u-v-t
        # Iterate over middle edge (u,v)
        for u, v in edge_set:
            # s in N(u), t in N(v)
            # s != v, t != u
            # s != t
            # deg(s) == deg(t)
            for s in neighbors[u]:
                if s == v:
                    continue
                for t in neighbors[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if degs[s] == degs[t]:
                        return False
        return True
    
    # Initial state
    degs = get_degrees(edges)
    neighbors = get_neighbors(edges)
    
    # Verify initial state (should be valid)
    if not is_valid(edges, degs, neighbors):
        # Fallback to empty or star if K78 is invalid (it shouldn't be)
        return [[0, v] for v in range(1, n)]
    
    best_edges = edges
    best_count = len(edges)
    
    # Local Search
    # We try to add edges. If we can't, we might try to swap.
    # But adding is strictly better for score.
    
    # Pre-compute all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
            
    # List of missing edges
    missing = [e for e in all_edges if e not in edges]
    
    while time.time() < deadline:
        random.shuffle(missing)
        improved = False
        for e in missing:
            u, v = e
            if (u, v) in edges:
                continue
            # Try adding
            new_edges = edges | {(u, v)}
            new_degs = degs[:]
            new_degs[u] += 1
            new_degs[v] += 1
            new_neighbors = [nb.copy() for nb in neighbors]
            new_neighbors[u].add(v)
            new_neighbors[v].add(u)
            
            if is_valid(new_edges, new_degs, new_neighbors):
                # Accept
                edges = new_edges
                degs = new_degs
                neighbors = new_neighbors
                missing.remove(e)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges.copy()
                improved = True
                if time.time() > deadline - 5:
                    break
        
        if not improved:
            # Try some swaps? Or just restart?
            # Given the structure, restarts might not help much if K78 is a local optimum.
            # But let's try a few random restarts from K78 with different orderings.
            # Or just break if we've tried all missing edges.
            # Since we shuffle, we are trying all.
            # If no edge can be added, we are stuck.
            
            # Try a random swap: remove one edge, add another?
            # This is complex to check efficiently.
            # Let's just try to restart the process with a slightly perturbed K78?
            # No, K78 is deterministic.
            
            # If we can't add any edge, we return best.
            break
            
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(len(res))
