import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_violations(adj):
        degs = [len(adj[i]) for i in range(n)]
        groups = {}
        for i in range(n):
            d = degs[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)
        
        violations = []
        for d, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                s = group[i]
                for j in range(i+1, len(group)):
                    t = group[j]
                    # Check if path s-u-v-t exists
                    # u in N(s), v in N(t), (u,v) in E, u!=v, u!=t, v!=s
                    found = False
                    for u in adj[s]:
                        if u == t: continue
                        for v in adj[t]:
                            if v == s: continue
                            if u == v: continue
                            if v in adj[u]:
                                violations.append((s, u, v, t))
                                found = True
                                break
                        if found: break
                    if found: break
        return violations

    def is_valid(adj):
        return len(get_violations(adj)) == 0

    best_edge_count = 0
    best_adj = None
    
    # Run multiple trials of the greedy removal
    for trial in range(2000):
        if time.time() > deadline - 5:
            break
            
        random.seed(trial)
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        removed_edges = []
        
        while True:
            if time.time() > deadline - 5:
                break
            violations = get_violations(adj)
            if not violations:
                break
            
            # Count edge frequencies in violations
            edge_counts = {}
            for s, u, v, t in violations:
                e1 = (min(s, u), max(s, u))
                e2 = (min(u, v), max(u, v))
                e3 = (min(v, t), max(v, t))
                
                for e in [e1, e2, e3]:
                    edge_counts[e] = edge_counts.get(e, 0) + 1
            
            if not edge_counts:
                break
                
            # Find max count
            max_count = max(edge_counts.values())
            candidates = [e for e, c in edge_counts.items() if c == max_count]
            
            # Randomly pick one candidate
            edge_to_remove = random.choice(candidates)
            
            u, v = edge_to_remove
            adj[u].discard(v)
            adj[v].discard(u)
            removed_edges.append(edge_to_remove)
            
        if is_valid(adj):
            count = 105 - len(removed_edges)
            if count > best_edge_count:
                best_edge_count = count
                best_adj = [set(adj[i]) for i in range(n)]
                
            # Try to add back edges
            # Iterate over missing edges
            missing = [e for e in all_edges if e not in [(min(u,v), max(u,v)) for u in range(n) for v in adj[u]]]
            # Simpler: iterate all_edges, check if in adj
            random.shuffle(missing)
            for edge in missing:
                if time.time() > deadline - 5:
                    break
                u, v = edge
                adj[u].add(v)
                adj[v].add(u)
                if is_valid(adj):
                    # Keep it
                    pass
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
            # Update best if improved
            final_count = sum(len(adj[i]) for i in range(n)) // 2
            if final_count > best_edge_count:
                best_edge_count = final_count
                best_adj = [set(adj[i]) for i in range(n)]

    if best_adj is None:
        # Fallback: return empty graph or something valid
        # Empty graph is valid (0 edges)
        best_adj = [set() for _ in range(n)]
        
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
                
    return edges
