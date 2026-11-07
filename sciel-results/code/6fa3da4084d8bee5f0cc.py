import time
import random
from itertools import combinations

def check_constraint(edges, n):
    """
    Check if no simple 3-edge path s-u-v-t has endpoints s, t with equal degree.
    Returns True if valid.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    # Group vertices by degree
    deg_groups = {}
    for i in range(n):
        if deg[i] not in deg_groups:
            deg_groups[deg[i]] = []
        deg_groups[deg[i]].append(i)
    
    # Check each group
    for d, vertices in deg_groups.items():
        if len(vertices) < 2:
            continue
        for s_idx in range(len(vertices)):
            for t_idx in range(s_idx + 1, len(vertices)):
                s = vertices[s_idx]
                t = vertices[t_idx]
                
                # Check if simple path s-u-v-t exists
                # u in adj[s], u != t
                # v in adj[t], v != s, v != u
                # (u, v) in edges
                
                found = False
                for u in adj[s]:
                    if u == t:
                        continue
                    # v must be in adj[u] and adj[t]
                    # and v != s, v != u
                    # adj[u] and adj[t] are sets
                    common = adj[u] & adj[t]
                    # Remove s and u from common
                    # v cannot be s (path s-u-s-t not simple)
                    # v cannot be u (no loops, so u not in adj[u])
                    # v cannot be t? t is in adj[t], but v is in adj[t].
                    # If v == t, path s-u-t-t? No, v is vertex.
                    # Path s-u-v-t. v != t is required for simple path?
                    # If v == t, path s-u-t. Length 2.
                    # We are looking for length 3.
                    # So v != t.
                    # Also u != t (checked).
                    # s != u (no loops).
                    # s != t (distinct in group).
                    
                    # So we need v in (adj[u] & adj[t]) such that v != s and v != t.
                    # Note: u is not in adj[u], so v != u is automatic if v in adj[u].
                    # But wait, if v=u, then u in adj[u] is false.
                    
                    for v in common:
                        if v == s or v == t:
                            continue
                        # Check if (u,v) is an edge?
                        # v in adj[u] implies (u,v) is an edge.
                        # v in adj[t] implies (v,t) is an edge.
                        # u in adj[s] implies (s,u) is an edge.
                        # So path s-u-v-t exists.
                        found = True
                        break
                    if found:
                        break
                if found:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 15
    
    # Initialize with K_{7,8}
    edges = []
    adj = [set() for _ in range(n)]
    deg = [0] * n
    
    for i in range(7):
        for j in range(7, 15):
            edges.append((i, j))
            adj[i].add(j)
            adj[j].add(i)
            deg[i] += 1
            deg[j] += 1
            
    best_edges = list(edges)
    best_count = len(edges)
    
    # Fast validity check using current adj and deg
    def is_valid(adj, deg):
        deg_groups = {}
        for i in range(n):
            if deg[i] not in deg_groups:
                deg_groups[deg[i]] = []
            deg_groups[deg[i]].append(i)
        
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for s_idx in range(len(vertices)):
                for t_idx in range(s_idx + 1, len(vertices)):
                    s = vertices[s_idx]
                    t = vertices[t_idx]
                    
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        common = adj[u] & adj[t]
                        for v in common:
                            if v != s and v != t:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True

    # Simulated Annealing
    rng = random.Random(42)
    current_edges = list(best_edges)
    current_adj = [set(s) for s in best_adj] if 'best_adj' in locals() else [set() for _ in range(n)]
    # Rebuild adj from edges
    current_adj = [set() for _ in range(n)]
    current_deg = [0] * n
    for u, v in current_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_deg[u] += 1
        current_deg[v] += 1
        
    current_score = len(current_edges)
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 5:
        # Pick a random move: Add, Remove, Swap
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        new_edges = list(current_edges)
        new_adj = [set(s) for s in current_adj]
        new_deg = list(current_deg)
        
        if move_type == 'add':
            # Pick random missing edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v or v in new_adj[u]:
                continue
            new_adj[u].add(v)
            new_adj[v].add(u)
            new_deg[u] += 1
            new_deg[v] += 1
            new_edges.append((u, v))
            
        elif move_type == 'remove':
            if not current_edges:
                continue
            idx = rng.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_deg[u] -= 1
            new_deg[v] -= 1
            new_edges.pop(idx)
            
        elif move_type == 'swap':
            if not current_edges:
                continue
            # Remove random edge
            idx = rng.randint(0, len(current_edges) - 1)
            u1, v1 = current_edges[idx]
            new_adj[u1].remove(v1)
            new_adj[v1].remove(u1)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            new_edges.pop(idx)
            
            # Add random missing edge
            u2 = rng.randint(0, n - 1)
            v2 = rng.randint(0, n - 1)
            if u2 == v2 or v2 in new_adj[u2]:
                # Retry add
                for _ in range(10):
                    u2 = rng.randint(0, n - 1)
                    v2 = rng.randint(0, n - 1)
                    if u2 != v2 and v2 not in new_adj[u2]:
                        break
                else:
                    # Revert remove
                    new_adj[u1].add(v1)
                    new_adj[v1].add(u1)
                    new_deg[u1] += 1
                    new_deg[v1] += 1
                    new_edges.append((u1, v1))
                    continue
            
            new_adj[u2].add(v2)
            new_adj[v2].add(u2)
            new_deg[u2] += 1
            new_deg[v2] += 1
            new_edges.append((u2, v2))
            
        # Check validity
        valid = is_valid(new_adj, new_deg)
        new_score = len(new_edges) if valid else -1000
        
        if new_score > current_score:
            current_edges = new_edges
            current_adj = new_adj
            current_deg = new_deg
            current_score = new_score
            if current_score > best_count:
                best_count = current_score
                best_edges = list(current_edges)
        else:
            # Accept with probability
            delta = new_score - current_score
            if delta < 0:
                prob = pow(2.71828, delta / T)
                if rng.random() < prob:
                    current_edges = new_edges
                    current_adj = new_adj
                    current_deg = new_deg
                    current_score = new_score
            else:
                current_edges = new_edges
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final result formatting
    result = []
    seen = set()
    for u, v in best_edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
            
    return result
