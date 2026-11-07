import time
import random

def _is_valid(edges, n):
    """Check no 3-edge path s-u-v-t (s!=t) has deg(s)==deg(t)."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Check all edges (u,v)
    for (u, v) in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _find_violation(edges, n):
    """Find one violating path or return None."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
    for (u, v) in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return (s, u, v, t)
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin
    n = 15
    best_edges = []
    
    # Try multiple random restarts
    for seed in range(100):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # 1. Start with a dense random graph
        edges = []
        edge_set = set()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.85:  # High density start
                    edges.append([u, v])
                    edge_set.add((u, v))
        
        # 2. Repair phase: remove edges to fix violations
        # We want to remove edges that are in many violations.
        # Simple approach: while invalid, remove a random edge from a violation.
        max_repair_steps = 200
        for _ in range(max_repair_steps):
            if time.time() > deadline:
                break
            viol = _find_violation(edges, n)
            if viol is None:
                break
            s, u, v, t = viol
            # The path is s-u-v-t. Edges are (s,u), (u,v), (v,t).
            # We can remove any of these, or edges incident to s or t to change degrees.
            # Removing (u,v) breaks this specific path but might not fix degree equality.
            # Changing degrees of s or t is more robust.
            # Let's try removing a random edge incident to s or t, or the path edges.
            candidates_to_remove = set()
            candidates_to_remove.add((s, u) if s < u else (u, s))
            candidates_to_remove.add((u, v) if u < v else (v, u))
            candidates_to_remove.add((v, t) if v < t else (t, v))
            
            # Also consider removing other edges incident to s or t
            for e in edges:
                a, b = e
                if a == s or b == s or a == t or b == t:
                    candidates_to_remove.add((a, b) if a < b else (b, a))
            
            if not candidates_to_remove:
                break
                
            # Pick a random candidate to remove
            rem_u, rem_v = random.choice(list(candidates_to_remove))
            rem_key = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
            
            if rem_key in edge_set:
                edge_set.remove(rem_key)
                # Remove from edges list
                for i, e in enumerate(edges):
                    if (e[0], e[1]) == rem_key or (e[1], e[0]) == rem_key:
                        edges.pop(i)
                        break
        
        # 3. Expansion phase: add edges greedily
        # Generate all missing edges
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    missing.append((u, v))
        
        # Shuffle to avoid bias
        random.shuffle(missing)
        
        for (u, v) in missing:
            if time.time() > deadline:
                break
            # Try adding
            new_edges = edges + [[u, v]]
            if _is_valid(new_edges, n):
                edges.append([u, v])
                edge_set.add((u, v))
        
        # Update best
        if len(edges) > len(best_edges):
            best_edges = edges[:]
            
    # Final check and return
    if not _is_valid(best_edges, n):
        # Fallback: empty graph is always valid
        return []
        
    return [[int(u), int(v)] for u, v in best_edges]
