import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 15
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid(edges_set, degrees):
        # Check if any pair with equal degree has distance 3
        # dist(s, t) == 3 iff exists u, v such that (s,u), (u,v), (v,t) in edges
        # and s != t, s != u, s != v, s != t, u != v, u != t, v != t?
        # Simple path s-u-v-t implies s,u,v,t distinct.
        
        # Group vertices by degree
        deg_to_verts = {}
        for i in range(n):
            d = degrees[i]
            if d not in deg_to_verts:
                deg_to_verts[d] = []
            deg_to_verts[d].append(i)
            
        # For each group with size > 1, check pairs
        for d, verts in deg_to_verts.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                s = verts[i]
                for j in range(i + 1, len(verts)):
                    t = verts[j]
                    # Check if dist(s, t) == 3
                    # Neighbors of s
                    ns = [u for u in range(n) if (min(s,u), max(s,u)) in edges_set]
                    # Neighbors of t
                    nt = [v for v in range(n) if (min(t,v), max(t,v)) in edges_set]
                    
                    # Check if any u in ns, v in nt such that (u,v) in edges_set
                    # and u != t, v != s, u != v
                    found = False
                    for u in ns:
                        if u == t: continue
                        for v in nt:
                            if v == s or v == u: continue
                            if (min(u,v), max(u,v)) in edges_set:
                                found = True
                                break
                        if found: break
                    if found:
                        return False
        return True

    def get_degrees(edges_set):
        degs = [0] * n
        for u, v in edges_set:
            degs[u] += 1
            degs[v] += 1
        return degs

    best_edges = set()
    best_count = 0
    
    # Greedy construction
    # Shuffle edges to allow different runs? No, deterministic is better for reproducibility, but random restarts help escape local optima.
    # Let's do a few random restarts.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        edges_set = set()
        degrees = [0] * n
        
        # Random order of edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
                
            # Try adding edge (u, v)
            new_edges = edges_set | {(u, v)}
            new_degrees = degrees[:]
            new_degrees[u] += 1
            new_degrees[v] += 1
            
            if is_valid(new_edges, new_degrees):
                edges_set = new_edges
                degrees = new_degrees
        
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = edges_set
            
        # Try local search: swap edges
        # If we have time, try to improve best_edges
        if time.time() > deadline - 10:
            break
            
        current_edges = best_edges
        current_degrees = get_degrees(current_edges)
        
        # Try adding an edge not in set
        # Or swapping an edge out for two in?
        # Simple: try adding any missing edge
        missing = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            new_edges = current_edges | {(u, v)}
            new_degrees = current_degrees[:]
            new_degrees[u] += 1
            new_degrees[v] += 1
            if is_valid(new_edges, new_degrees):
                current_edges = new_edges
                current_degrees = new_degrees
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = current_edges

    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
