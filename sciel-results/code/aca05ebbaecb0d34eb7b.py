import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n = 18
    
    # Use bitmask representation for adjacency
    # adj[i] is an integer where bit j is set if edge (i,j) exists
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if (adj[u] >> v) & 1:
            return True
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        common_list = [j for j in range(n) if (common >> j) & 1]
        for idx, w in enumerate(common_list):
            for x in common_list[idx+1:]:
                if (adj[w] >> x) & 1:
                    return False
        return True
    
    def largest_triangle_free_subset(adj):
        """Find the largest subset of vertices inducing a triangle-free subgraph.
        Uses branch and bound with bitmasks."""
        best = [0]
        
        # Precompute: for each vertex v, the bitmask of its neighbors
        # For a set S (bitmask), v can be added to S iff no two neighbors of v in S are adjacent.
        # i.e., for all i, j in S, if i and j are both neighbors of v, then i and j are not adjacent.
        
        def bnb(S_mask, candidates_mask, depth):
            if time.time() > deadline:
                return
            current_size = bin(S_mask).count('1')
            
            # Find all addable vertices
            addable = []
            for v in range(n):
                if (candidates_mask >> v) & 1:
                    NSv = adj[v] & S_mask
                    # Check if NSv is an independent set
                    ok = True
                    # Iterate through bits in NSv
                    NSv_list = [x for x in range(n) if (NSv >> x) & 1]
                    for ai, a in enumerate(NSv_list):
                        for b in NSv_list[ai+1:]:
                            if (adj[a] >> b) & 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        addable.append(v)
            
            if not addable:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Try adding each addable vertex
            for v in addable:
                new_S = S_mask | (1 << v)
                new_cands = candidates_mask & ~(1 << v)
                bnb(new_S, new_cands, depth + 1)
                if time.time() > deadline:
                    return
                # Pruning: if current_size + len(addable) <= best[0], we could prune
                # But we already recursed, so skip for now
            
            # Also consider not adding any more vertices (already handled by the base case)
        
        full_mask = (1 << n) - 1
        bnb(0, full_mask, 0)
        return best[0]
    
    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Start with a random K4-free graph
    # Use a random initial graph and try to add edges while keeping K4-free
    random.seed(42)
    adj = [0] * n
    
    # Add random edges while keeping K4-free
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    current_score = largest_triangle_free_subset(adj)
    best_adj = adj[:]
    best_score = current_score
    
    print(f"Initial score: {best_score}")
    
    # Local search: try to improve
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, best_score={best_score}, time={time.time()-start_time:.1f}s")
        
        # Try random edge additions/removals
        made_change = False
        
        # Try adding random edges
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if not (adj[u] >> v) & 1:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    made_change = True
                    break
        
        # Try removing random edges  
        for _ in range(30):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                made_change = True
                break
        
        if not made_change:
            # Reset to best and randomize more
            adj = best_adj[:]
            continue
        
        # Evaluate new score
        new_score = largest_triangle_free_subset(adj)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = adj[:]
            print(f"Improved! New best score: {best_score} at iteration {iteration}")
        elif new_score == best_score and random.random() < 0.1:
            # Sometimes keep equal-score states for diversity
            pass
        else:
            # Revert if worse
            adj = best_adj[:]
    
    # Return best found
    edges = get_edge_list(best_adj)
    return edges
