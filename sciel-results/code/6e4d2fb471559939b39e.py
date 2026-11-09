"""Improved program: greedy edge removal from complete graph to satisfy path constraint."""
import time
import random

# EVOLVE-BLOCK-START

def _violations(adj, n):
    """Return list of violating paths (s,u,v,t) where deg(s)==deg(t)."""
    deg = [0]*n
    for i in range(n):
        deg[i] = adj[i].__len__() if isinstance(adj[i], set) else len(adj[i])
    
    viol = []
    for u in range(n):
        for v in list(adj[u]):
            if u >= v:
                continue
            # For each edge (u,v), find s in N(u)\{v} and t in N(v)\{u} with s!=t
            nu = list(adj[u])
            nv = list(adj[v])
            # s must be in N(u), s != v, s != u (s != u always since no loops)
            # t must be in N(v), t != u, t != v
            for s in nu:
                if s == v:
                    continue
                for t in nv:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        viol.append((s, u, v, t))
    return viol

def _violations_count_fast(adj, n):
    """Return count of violations quickly."""
    deg = [len(adj[i]) for i in range(n)]
    count = 0
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        count += 1
    return count

def _is_valid(adj, n):
    """Check if graph satisfies the constraint."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
    return True

def construct():
    n = 15
    deadline = time.time() + 75  # 75s out of 90s budget
    
    # Start with complete graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    best_edges = []
    best_count = 0
    
    # Greedy removal: repeatedly remove the edge that causes the most violations
    # or use a smarter heuristic
    
    while time.time() < deadline:
        # Compute violations and find best edge to remove
        deg = [len(adj[i]) for i in range(n)]
        
        # For each edge, compute how many violations it participates in
        edge_viol = {}
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                key = (u, v)
                edge_viol[key] = 0
        
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    ds = deg[s]
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[t] == ds:
                            # This violation involves edge (u,v)
                            edge_viol[(u,v)] += 1
                            # Also, the endpoints s and t are equal degree
                            # The edge (u,v) is the middle edge
        
        if not any(edge_viol.values()):
            break
        
        # Find the edge with maximum violations
        max_edge = max(edge_viol.items(), key=lambda x: x[1])
        u, v = max_edge[0]
        
        # Remove this edge
        adj[u].discard(v)
        adj[v].discard(u)
        
        # Check if valid
        if _is_valid(adj, n):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
        
        # If no more violations, we're done
        if not any(edge_viol.values()):
            break
    
    # If we never found a valid graph, try a different approach
    if not best_edges:
        # Try random restarts with different initial graphs
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Start with random graph
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Greedy removal
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                deg = [len(adj[i]) for i in range(n)]
                
                edge_viol = {}
                for u in range(n):
                    for v in adj[u]:
                        if u >= v:
                            continue
                        edge_viol[(u,v)] = 0
                
                has_viol = False
                for u in range(n):
                    for v in adj[u]:
                        if u >= v:
                            continue
                        for s in adj[u]:
                            if s == v:
                                continue
                            ds = deg[s]
                            for t in adj[v]:
                                if t == u or t == s:
                                    continue
                                if deg[t] == ds:
                                    edge_viol[(u,v)] += 1
                                    has_viol = True
                
                if not has_viol:
                    break
                
                max_edge = max(edge_viol.items(), key=lambda x: x[1])
                u, v = max_edge[0]
                adj[u].discard(v)
                adj[v].discard(u)
            
            if _is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                count = len(edges)
                if count > best_count:
                    best_count = count
                    best_edges = edges
    
    # Ensure we return at least something valid
    if not best_edges:
        # Return a star (which is always valid since it's a tree, no 3-edge paths with equal degree endpoints? Let me check)
        # Actually a star: center 0, leaves 1-14. Path s-u-v-t: u=0, v=leaf, s=another leaf, t=0? No, t must be in N(v).
        # In a star, N(0) = {1..14}, N(leaf) = {0}. So for edge (0, leaf), s in N(0)\{leaf}, t in N(leaf)\{0} = empty. So no violations.
        best_edges = [[0, v] for v in range(1, n)]
    
    return best_edges

# EVOLVE-BLOCK-END
