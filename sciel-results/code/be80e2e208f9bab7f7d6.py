"""Rewritten program: construct a C4-free graph with no independent 11-set, maximizing n."""
import time
import random
import itertools

def _deadline_ok(start, limit=110.0):
    return (time.time() - start) < limit

def construct():
    start = time.time()
    # We will try to build a graph with n vertices, C4-free, alpha <= 10.
    # Strategy: start from a known small C4-free graph and add vertices while maintaining constraints.
    # Or use a greedy random construction with backtracking.
    
    # A simple approach: use a greedy algorithm to add vertices one by one.
    # For each new vertex, connect it to a subset of existing vertices such that:
    # 1. No C4 is formed (no two neighbors of the new vertex share a neighbor)
    # 2. The independence number remains <= 10
    
    # Actually, let's try a different approach: use a known construction or random search.
    
    # Let's try random construction with verification.
    # For n up to 39, we can try random graphs and check.
    
    # But random graphs are unlikely to be C4-free. Let's use a structured approach.
    
    # Approach: Start with a complete bipartite graph? No, that has C4s.
    # Start with a cycle? Cycles can have C4s.
    
    # Let's use a greedy algorithm:
    # 1. Start with an empty graph.
    # 2. Add vertices one by one.
    # 3. For each new vertex, choose a set of neighbors from existing vertices such that:
    #    - No two neighbors have a common neighbor (to avoid C4)
    #    - The independence number doesn't exceed 10
    
    # This is complex. Let's try a simpler approach: use a known C4-free graph construction.
    
    # Known: The polarity graph of a projective plane is C4-free.
    # For q=5, the polarity graph has 31 vertices and is C4-free.
    # We can try to extend this to more vertices.
    
    # Let's implement a greedy random search:
    # 1. Start with a small C4-free graph.
    # 2. Try to add vertices by connecting them to a random subset of existing vertices.
    # 3. Check if the graph remains C4-free and alpha <= 10.
    
    # For efficiency, let's use a fixed seed for reproducibility.
    random.seed(42)
    
    # Start with a triangle (3 vertices, C4-free, alpha=1)
    n = 3
    edges = [[0, 1], [0, 2], [1, 2]]
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Try to add vertices up to 39
    target_n = 39
    max_attempts = 10000
    
    for target in range(4, target_n + 1):
        if not _deadline_ok(start):
            break
        added = False
        for attempt in range(max_attempts):
            if not _deadline_ok(start):
                break
            # Try to add vertex 'target' with random neighbors
            # Choose a random subset of existing vertices as neighbors
            # To avoid C4, the neighbors must form an independent set in the "common neighbor" sense
            # i.e., no two neighbors share a common neighbor
            
            # Generate a random subset
            k = random.randint(1, min(target, 10))  # Limit degree for simplicity
            if k > target:
                k = target
            neighbors = set(random.sample(range(target), k))
            
            # Check C4-freedom: for any two neighbors u, v of the new vertex,
            # they must not share a common neighbor
            c4_free = True
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    u, v = neighbor_list[i], neighbor_list[j]
                    if adj[u] & adj[v]:
                        c4_free = False
                        break
                if not c4_free:
                    break
            
            if not c4_free:
                continue
            
            # Check independence number: adding a vertex with neighbors S
            # The new vertex is not in any independent set that contains any of its neighbors.
            # We need to check if alpha remains <= 10.
            # This is expensive, so let's do a quick check:
            # The new vertex can be added to independent sets that don't contain any of its neighbors.
            # Let's compute the maximum independent set size in the new graph.
            
            # For efficiency, let's only do a full check every few attempts
            if attempt % 10 == 0:
                # Compute alpha of the new graph
                new_n = target + 1
                new_adj = [set(s) for s in adj]
                new_adj.append(set())
                for u in neighbors:
                    new_adj[u].add(target)
                    new_adj[target].add(u)
                
                alpha = _max_independent_set(new_n, new_adj)
                if alpha > 10:
                    continue
            
            # Add the vertex
            adj.append(set(neighbors))
            for u in neighbors:
                adj[u].add(target)
            edges.append([target, min(target, 0)])  # Placeholder, we'll fix edges later
            added = True
            break
        
        if not added:
            # Couldn't add this vertex, try a different strategy or stop
            # For now, break
            break
    
    # Reconstruct edges from adj
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Verify
    n_final = len(adj)
    if _is_c4_free(n_final, adj) and _max_independent_set(n_final, adj) <= 10:
        return {'n': n_final, 'edges': edges}
    
    # Fallback: return the initial triangle
    return {'n': 3, 'edges': [[0, 1], [0, 2], [1, 2]]}

def _is_c4_free(n, adj):
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u] & adj[v]:
                    return False
    return True

def _max_independent_set(n, adj):
    # Simple branch and bound
    best = 0
    def dfs(current_set, remaining):
        nonlocal best
        if not remaining:
            best = max(best, len(current_set))
            return
        # Prune
        if len(current_set) + len(remaining) <= best:
            return
        # Choose a vertex
        v = next(iter(remaining))
        # Include v
        new_remaining = {u for u in remaining if u != v and u not in adj[v]}
        dfs(current_set | {v}, new_remaining)
        # Exclude v
        remaining.remove(v)
        dfs(current_set, remaining)
        remaining.add(v)
    
    dfs(set(), set(range(n)))
    return best

# Test
if __name__ == '__main__':
    result = construct()
    print(result['n'], len(result['edges']))
