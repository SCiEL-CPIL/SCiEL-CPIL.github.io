import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid(edges):
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for u in range(n):
            adj_u = adj[u]
            for v in adj_u:
                adj_v = adj[v]
                for s in adj_u:
                    if s == v:
                        continue
                    for t in adj_v:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    # Initialize
    # Try a few random starts
    best_edges = []
    best_score = 0
    
    for _ in range(10):
        if time.time() > deadline - 10:
            break
            
        # Random graph
        p = random.uniform(0.4, 0.8)
        current_edges = [e for e in all_edges if random.random() < p]
        
        # Repair: Remove edges until valid
        while current_edges and not is_valid(current_edges):
            # Find a violating path and remove one of its edges
            deg = get_degrees(current_edges)
            adj = [set() for _ in range(n)]
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            removed = False
            for u in range(n):
                if removed: break
                adj_u = adj[u]
                for v in adj_u:
                    if removed: break
                    adj_v = adj[v]
                    for s in adj_u:
                        if removed or s == v:
                            continue
                        for t in adj_v:
                            if removed or t == u or t == s:
                                continue
                            if deg[s] == deg[t]:
                                # Remove a random edge from the path
                                path_edges = [(s,u), (u,v), (v,t)]
                                # Convert to canonical form
                                path_edges_canon = []
                                for a,b in path_edges:
                                    if a < b: path_edges_canon.append((a,b))
                                    else: path_edges_canon.append((b,a))
                                
                                # Remove one
                                rem_e = random.choice(path_edges_canon)
                                current_edges = [e for e in current_edges if (min(e[0],e[1]), max(e[0],e[1])) != rem_e]
                                removed = True
                                break
            if not removed:
                # Should not happen if is_valid is correct, but safety
                break

        if len(current_edges) > best_score:
            best_score = len(current_edges)
            best_edges = current_edges[:]
            
        # Local Search on current_edges
        while time.time() < deadline - 5:
            improved = False
            # Try adding edges
            present = set()
            for u, v in current_edges:
                present.add((min(u,v), max(u,v)))
                
            missing = [e for e in all_edges if (min(e[0],e[1]), max(e[0],e[1])) not in present]
            random.shuffle(missing)
            
            for e in missing:
                if time.time() > deadline - 5:
                    break
                new_edges = current_edges + [e]
                if is_valid(new_edges):
                    current_edges = new_edges
                    improved = True
                    if len(current_edges) > best_score:
                        best_score = len(current_edges)
                        best_edges = current_edges[:]
                        
            if not improved:
                # Try swaps
                if len(current_edges) < 1:
                    break
                present = set()
                for u, v in current_edges:
                    present.add((min(u,v), max(u,v)))
                
                missing = [e for e in all_edges if (min(e[0],e[1]), max(e[0],e[1])) not in present]
                random.shuffle(missing)
                
                swapped = False
                for e_add in missing:
                    if time.time() > deadline - 5:
                        break
                    for i in range(len(current_edges)):
                        e_rem = current_edges[i]
                        new_edges = current_edges.copy()
                        new_edges.pop(i)
                        new_edges.append(e_add)
                        if is_valid(new_edges):
                            current_edges = new_edges
                            swapped = True
                            if len(current_edges) > best_score:
                                best_score = len(current_edges)
                                best_edges = current_edges[:]
                            break
                    if swapped:
                        break
                if not swapped:
                    break

    # Final check and type conversion
    final_edges = []
    for u, v in best_edges:
        final_edges.append([int(u), int(v)])
        
    return final_edges
