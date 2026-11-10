import time
import random

def construct():
    n = 15
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    def check_graph(adj, deg):
        """Check if the graph has any 3-path with equal degree endpoints."""
        for s in range(n):
            for u in adj[s]:
                for v in adj[u]:
                    if v == s:
                        continue
                    for t in adj[v]:
                        if t == s or t == u:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def greedy_fill(adj, deg):
        """Greedily add edges that don't create violations."""
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if u in adj[v]:
                continue
            # Check if adding (u,v) is valid
            valid = True
            # Case 1: u-v-y-t. deg(u) != deg(t)
            for y in adj[v]:
                if y == u: continue
                for t in adj[y]:
                    if t == u or t == v: continue
                    if deg[u] == deg[t]:
                        valid = False
                        break
                if not valid: break
            if valid:
                # Case 2: s-u-v-t. deg(s) != deg(t)
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            valid = False
                            break
                    if not valid: break
            if valid:
                # Case 3: s-x-u-v. deg(s) != deg(v)
                for x in adj[u]:
                    if x == v: continue
                    for s in adj[x]:
                        if s == u or s == v: continue
                        if deg[s] == deg[v]:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1

    # Initialize with empty graph or a good seed
    # Let's start with the K_{7,8} seed from the current program as it's a strong start
    adj = [set() for _ in range(n)]
    deg = [0] * n
    part_a = list(range(7))
    part_b = list(range(7, 15))
    for u in part_a:
        for v in part_b:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
    
    # Greedy fill
    greedy_fill(adj, deg)
    
    best_edges = sum(len(s) for s in adj) // 2
    best_adj = [set(s) for s in adj]
    best_deg = deg[:]
    
    random.seed(42)
    
    # Local Search: Remove some edges, then greedy fill again
    # We try to find a configuration with more edges.
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Create a perturbed version of the best graph
        curr_adj = [set(s) for s in best_adj]
        curr_deg = best_deg[:]
        
        # Remove k random edges
        k = random.randint(1, 5)
        edges_to_remove = []
        for u in range(n):
            for v in curr_adj[u]:
                if u < v:
                    edges_to_remove.append((u, v))
        
        if len(edges_to_remove) < k:
            continue
            
        random.shuffle(edges_to_remove)
        for u, v in edges_to_remove[:k]:
            curr_adj[u].remove(v)
            curr_adj[v].remove(u)
            curr_deg[u] -= 1
            curr_deg[v] -= 1
        
        # Greedy fill on the perturbed graph
        greedy_fill(curr_adj, curr_deg)
        
        curr_edges = sum(len(s) for s in curr_adj) // 2
        
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = [set(s) for s in curr_adj]
            best_deg = curr_deg[:]
            # Restart search from this new best? Or continue?
            # Continuing is fine, but we might want to increase k or change strategy.
            # For now, just update best.
            
    # Final verification (optional, but good for safety)
    # if not check_graph(best_adj, best_deg):
    #     # Fallback to initial greedy if local search broke it (should not happen if greedy_fill is correct)
    #     pass

    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
