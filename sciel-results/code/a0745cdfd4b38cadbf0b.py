import random
import time

def construct():
        n = 15
        # Start with empty graph
        edges = []
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        def is_valid_addition(u, v):
            # Check if adding (u, v) violates constraint
            # 1. Check paths involving (u, v) as the middle edge
            #    s - u - v - t
            #    s in N(u)\{v}, t in N(v)\{u}
            #    Need deg(s) != deg(t)
            
            # Note: deg(u) and deg(v) will increase by 1 after addition.
            # But the constraint checks degrees in the FULL graph.
            # So we should check with deg(u)+1 and deg(v)+1?
            # No, the path is s-u-v-t. s and t are endpoints.
            # Their degrees are deg(s) and deg(t).
            # u and v are internal. Their degrees don't appear in the equality check deg(s) != deg(t).
            # However, adding (u,v) changes deg(u) and deg(v).
            # This might affect validity of OTHER edges.
            
            # First, check if (u,v) itself causes a violation with existing neighbors.
            for s in adj[u]:
                if s == v: continue
                for t in adj[v]:
                    if t == u: continue
                    if s == t: continue # Path must be simple? "simple three-edge path". 
                    # Usually simple path means distinct vertices. s-u-v-t implies s!=u, u!=v, v!=t.
                    # And s!=t? "s != t" is explicitly stated.
                    # Also s!=v, u!=t, s!=u are implied by simple path and no loops.
                    # If s == t, it's a cycle, not a simple path in the sense of distinct endpoints?
                    # The prompt says "simple three-edge path ... s != t".
                    # So if s == t, it's not a valid path to check? Or is it a cycle?
                    # "simple path" usually implies all vertices distinct.
                    # So s, u, v, t must be distinct.
                    if deg[s] == deg[t]:
                        return False
            
            # Second, check if adding (u,v) breaks existing edges.
            # Adding (u,v) increases deg(u) and deg(v).
            # For any existing edge (x, y), we need to re-check paths x-y-...
            # This is complex. 
            # Maybe it's safer to just do a full validation after each addition?
            # Full validation:
            # For every edge (a,b) in graph:
            #   for s in N(a)\{b}:
            #     for t in N(b)\{a}:
            #       if s != t and deg[s] == deg[t]: return False
            # Cost: E * N^2. 105 * 225 ~ 23k.
            # 105 additions * 23k = 2.5M. Very fast.
            
            return True
        
        def full_validation():
            for i in range(n):
                for j in range(n):
                    if i < j and j in adj[i]:
                        # Edge (i, j)
                        for s in adj[i]:
                            if s == j: continue
                            for t in adj[j]:
                                if t == i: continue
                                if s == t: continue
                                if deg[s] == deg[t]:
                                    return False
            return True

        # Greedy addition
        # Try all edges in random order?
        # Or try to maximize density.
        # Let's try adding edges in order of (u+v) or random.
        
        # To beat 56, we need a smarter approach or luck.
        # K_{7,8} is 56.
        # Let's try to construct a graph close to complete but with degree constraints.
        
        # Actually, let's just run the greedy with random restarts.
        best_edges = []
        start_time = time.time()
        
        for _ in range(100): # 100 restarts
            if time.time() - start_time > 50: break
            random.seed(_)
            adj = [set() for _ in range(n)]
            deg = [0] * n
            edges = []
            
            # Try adding edges in random order
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding is safe
                # We'll do a quick check on the new edge, then full validation if passed?
                # Or just full validation?
                
                # Add temporarily
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                
                if full_validation():
                    edges.append([u, v])
                else:
                    # Remove
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
            
            if len(edges) > len(best_edges):
                best_edges = [list(e) for e in edges]
                
        return best_edges
