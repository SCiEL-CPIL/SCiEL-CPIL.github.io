import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin
    
    # Helper to check C5 and Alpha
    def is_valid(n, edges):
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check C5
        # A C5 exists if there is a vertex v and two neighbors u, w of v
        # such that there is a path u-x-y-w of length 3 in G - {v}
        # which implies u and w are not adjacent in G-v? No, u and w can be adjacent?
        # If u and w are adjacent, u-x-y-w-v-u is not a simple cycle if u-w is an edge?
        # The cycle is v-u-x-y-w-v.
        # Vertices: v, u, x, y, w. Distinct.
        # Edges: (v,u), (u,x), (x,y), (y,w), (w,v).
        # We must ensure u, x, y, w are distinct and not v.
        # x, y must be distinct from u, w, v.
        
        for v in range(n):
            neighbors = list(adj[v])
            for i in range(len(neighbors)):
                u = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    # Check if path u-x-y-w exists in G - {v}
                    # x in N(u) \ {v, w}
                    # y in N(w) \ {v, u}
                    # x != y
                    # (x, y) in edges
                    found_path = False
                    for x in adj[u]:
                        if x == v or x == w:
                            continue
                        for y in adj[w]:
                            if y == v or y == u:
                                continue
                        if x == y:
                            continue
                        if y in adj[x]:
                            # Path u-x-y-w found
                            # Cycle v-u-x-y-w-v
                            # Check distinctness:
                            # v, u, x, y, w
                            # u, w are neighbors of v.
                            # x, y are not v.
                            # x != w, y != u.
                            # x != y.
                            # Are u, x, y, w distinct?
                            # u != w (i != j)
                            # x != u (x in N(u), simple graph)
                            # y != w (y in N(w))
                            # x != w (checked)
                            # y != u (checked)
                            # x != y (checked)
                            return False # Invalid (has C5)
        # Check Alpha
        # Greedy or simple check for independent set of size 9
        # Since n <= 35, we can use a simple backtracking or greedy heuristic
        # But for exact check, we need to be careful.
        # However, if we are searching, a fast heuristic check is better.
        # Let's use a simple greedy independent set finder to get a lower bound on alpha.
        # If lower bound >= 9, it's invalid.
        # If upper bound < 9, it's valid.
        # For exactness, we might need a proper check.
        # Given time, let's use a simple recursive check for alpha >= 9.
        
        def has_independent_set_of_size_k(n, adj, k):
            # Backtracking
            # This is exponential, but n=35, k=9.
            # Might be slow.
            # Let's try a greedy first.
            # If greedy finds 9, return True.
            # If not, we might miss it.
            # For the purpose of finding a valid graph, if we assume the graph is "random-like", 
            # greedy is often good.
            # But we need EXACT validity for the final answer.
            # The task says "both checked exactly".
            # So I must implement an exact check.
            
            # Use a simple DFS with pruning
            # State: current independent set, available vertices
            # This is too slow for n=35 in Python if not optimized.
            
            # Alternative: Use the fact that we are modifying a known valid graph.
            # If we start with 8 K4s (alpha=8) and make small changes, 
            # we can update alpha efficiently?
            
            # Let's stick to the C5 check being the main bottleneck for search.
            # For the final return, we will verify exactly.
            
            # For now, in the search loop, we'll use a heuristic alpha check.
            # Heuristic: Greedy maximal independent set.
            # If max_greedy >= 9, likely invalid.
            # If max_greedy < 9, might be valid.
            
            # Let's just return None for alpha check in the loop and do exact check at the end?
            # No, we need to prune the search.
            
            # Let's use a simple upper bound: 
            # If the graph is dense, alpha is small.
            
            # Actually, let's just implement the exact check for the final candidate.
            # In the search loop, we only check C5.
            # Then we verify the best C5-free candidate for Alpha.
            
            return False # Placeholder
        
        # If we reach here, no C5 found.
        return True

    # Search for n=33, 34, 35
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        while time.time() < deadline and not found:
            attempts += 1
            # Random graph generation
            # Start with 8 K4s and add random vertices/edges?
            # Or just random graph?
            # Random graph with p=0.5 is likely to have C5.
            # Let's try to perturb the 32-vertex graph.
            
            edges = list(best_edges)
            # Add vertices target_n - 32
            for new_v in range(32, target_n):
                # Connect new_v to random subset of existing vertices
                # To avoid C5, connect to few vertices?
                # Or connect to a clique?
                # Let's try connecting to a random vertex in each K4?
                # Or just random edges.
                num_edges_to_add = random.randint(2, 8)
                targets = random.sample(range(new_v), num_edges_to_add)
                for t in targets:
                    edges.append([new_v, t])
            
            # Check C5
            if is_valid(target_n, edges):
                # Verify Alpha exactly
                # Implement exact alpha check here
                # If valid, update best
                if check_alpha_exact(target_n, edges) < 9:
                    best_n = target_n
                    best_edges = edges
                    found = True
    
    return {'n': best_n, 'edges': best_edges}
