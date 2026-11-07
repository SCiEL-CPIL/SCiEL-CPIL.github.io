import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def is_valid_graph(adj):
        deg = get_degrees(adj)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    best_edges = set()
    best_count = 0
    
    # Strategy 1: K_{7,8} + Local Search
    def run_search(adj, edges):
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        missing = [e for e in all_pairs if e not in edges]
        current_count = len(edges)
        fail_count = 0
        
        while time.time() < deadline - 2:
            if not missing:
                break
            
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            adj[u].add(v)
            adj[v].add(u)
            edges.add((u, v))
            
            if is_valid_graph(adj):
                missing.remove((u, v))
                current_count += 1
                fail_count = 0
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(edges)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                edges.remove((u, v))
                fail_count += 1
                
                if fail_count > 50:
                    # Swap
                    if edges:
                        rem_u, rem_v = random.choice(list(edges))
                        adj[rem_u].remove(rem_v)
                        adj[rem_v].remove(rem_u)
                        edges.remove((rem_u, rem_v))
                        missing.append((rem_u, rem_v))
                        current_count -= 1
                        fail_count = 0
                        
                        # Try to add a different edge
                        if missing:
                            idx2 = random.randint(0, len(missing) - 1)
                            u2, v2 = missing[idx2]
                            adj[u2].add(v2)
                            adj[v2].add(u2)
                            edges.add((u2, v2))
                            if is_valid_graph(adj):
                                missing.remove((u2, v2))
                                current_count += 1
                                if current_count > best_count:
                                    best_count = current_count
                                    best_edges = set(edges)
                            else:
                                adj[u2].remove(v2)
                                adj[v2].remove(u2)
                                edges.remove((u2, v2))
                                missing.append((u2, v2))
                                # Revert removal
                                adj[rem_u].add(rem_v)
                                adj[rem_v].add(rem_u)
                                edges.add((rem_u, rem_v))
                                missing.remove((rem_u, rem_v))
                                current_count += 1

    # Init K7,8
    adj = [set() for _ in range(n)]
    edges = set()
    for i in range(7):
        for j in range(7, 15):
            adj[i].add(j)
            adj[j].add(i)
            edges.add((i, j))
            
    run_search(adj, edges)
    
    # Strategy 2: Start from K15, remove until valid, then add
    if time.time() < deadline - 10:
        adj2 = [set() for _ in range(n)]
        edges2 = set()
        for i in range(n):
            for j in range(i+1, n):
                adj2[i].add(j)
                adj2[j].add(i)
                edges2.add((i, j))
        
        # Remove until valid
        while not is_valid_graph(adj2) and time.time() < deadline - 5:
            # Find a violating edge and remove it?
            # Or just remove random edge
            u, v = random.choice(list(edges2))
            adj2[u].remove(v)
            adj2[v].remove(u)
            edges2.remove((u, v))
            
        if is_valid_graph(adj2):
            # Now try to add edges back
            run_search(adj2, edges2)
            
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
