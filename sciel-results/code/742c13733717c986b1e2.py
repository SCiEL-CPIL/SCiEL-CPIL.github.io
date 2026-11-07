import time
import random

def construct():
    n = 15
    deadline = time.time() + 85  # 85s safety margin
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def count_violations(adj):
        """Count number of violating paths s-u-v-t with deg(s)==deg(t)"""
        deg = get_degrees(adj)
        violations = 0
        for u in range(n):
            for v in adj[u]:
                if v < u:
                    continue
                # For edge (u,v), check all s in adj[u]\{v} and t in adj[v]\{u}
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        violations += 1
        return violations
    
    def is_valid(adj):
        return count_violations(adj) == 0
    
    def greedy_remove(adj, max_iters=10000):
        """Remove edges causing most violations until valid"""
        adj_set = [set(adj[i]) for i in range(n)]
        for i in range(n):
            for j in adj_set[i]:
                if j > i:
                    adj_set[j].add(i)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            deg = [len(adj_set[i]) for i in range(n)]
            # Find edge with most violations
            worst_edge = None
            worst_violations = 0
            for u in range(n):
                for v in adj_set[u]:
                    if v < u:
                        continue
                    violations = 0
                    for s in adj_set[u]:
                        if s == v:
                            continue
                        for t in adj_set[v]:
                            if t == u:
                                continue
                            if s == t:
                                continue
                            if deg[s] == deg[t]:
                                violations += 1
                    if violations > worst_violations:
                        worst_violations = violations
                        worst_edge = (u, v)
            
            if worst_violations == 0:
                break
            
            u, v = worst_edge
            adj_set[u].discard(v)
            adj_set[v].discard(u)
        
        return adj_set
    
    def try_add_edges(adj_set):
        """Try to add edges that don't create violations"""
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            deg = [len(adj_set[i]) for i in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    if v in adj_set[u]:
                        continue
                    # Check if adding (u,v) creates violations
                    new_violations = 0
                    for s in adj_set[u]:
                        for t in adj_set[v]:
                            if s == t:
                                continue
                            if deg[s] == deg[t]:
                                new_violations += 1
                    if new_violations == 0:
                        adj_set[u].add(v)
                        adj_set[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        improved = True
                        # Need to recheck since degrees changed
                        deg = [len(adj_set[i]) for i in range(n)]
                        break
                if improved:
                    break
        return adj_set
    
    def adj_to_edges(adj_set):
        edges = []
        for u in range(n):
            for v in adj_set[u]:
                if v > u:
                    edges.append([u, v])
        return edges
    
    best_edges = None
    best_count = 0
    
    # Multiple random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random initial graph
        adj_set = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj_set[u].add(v)
                    adj_set[v].add(u)
        
        # Greedy removal
        adj_set = greedy_remove(adj_set)
        
        # Try to add edges
        adj_set = try_add_edges(adj_set)
        
        # Final check
        if is_valid(adj_set):
            edges = adj_to_edges(adj_set)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
    
    # If no valid graph found, return star
    if best_edges is None:
        return [[0, v] for v in range(1, n)]
    
    return best_edges
