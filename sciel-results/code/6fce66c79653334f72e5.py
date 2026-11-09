import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s limit
    rng = random.Random(42)

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None  # duplicate
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    best = None
    best_e = 0

    def numpy_greedy_build(seed_rng):
        """Build a K2,3-free graph using vectorized NumPy common-neighbor checks."""
        adj = np.zeros((n, n), dtype=np.uint8)
        common = np.zeros((n, n), dtype=np.int16)
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        seed_rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if adj[u, v]:
                continue
            
            neighbors_u = np.where(adj[u] == 1)[0]
            neighbors_v = np.where(adj[v] == 1)[0]
            
            if len(neighbors_u) > 0:
                if np.max(common[neighbors_u, v]) > 1:
                    continue
            
            if len(neighbors_v) > 0:
                if np.max(common[u, neighbors_v]) > 1:
                    continue
            
            adj[u, v] = 1
            adj[v, u] = 1
            
            for w in neighbors_u:
                if w != v:
                    common[w, v] += 1
                    common[v, w] += 1
            
            for w in neighbors_v:
                if w != u:
                    common[u, w] += 1
                    common[w, u] += 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    def numpy_local_search(edges, max_iters=300):
        """Local search using NumPy. Includes swap moves to escape local optima."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        improved = True
        iters = 0
        current_edge_count = len(edges)
        
        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            
            # 1. Try to add an edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            rng.shuffle(non_edges)
            
            added = False
            for u, v in non_edges:
                neighbors_u = np.where(adj[u] == 1)[0]
                neighbors_v = np.where(adj[v] == 1)[0]
                
                valid = True
                if len(neighbors_u) > 0:
                    if np.max(common[neighbors_u, v]) > 1:
                        valid = False
                if valid and len(neighbors_v) > 0:
                    if np.max(common[u, neighbors_v]) > 1:
                        valid = False
                
                if valid:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    
                    for w in neighbors_u:
                        if w != v:
                            common[w, v] += 1
                            common[v, w] += 1
                    for w in neighbors_v:
                        if w != u:
                            common[u, w] += 1
                            common[w, u] += 1
                    
                    current_edge_count += 1
                    improved = True
                    added = True
                    break
            
            # 2. If no edge added, try a swap (remove 1, add 2)
            if not added and current_edge_count > 20:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if edge_list:
                    # Try a few random swaps
                    for _ in range(5):
                        u_rem, v_rem = rng.choice(edge_list)
                        
                        # Save state
                        adj_save = adj.copy()
                        common_save = common.copy()
                        count_save = current_edge_count
                        
                        # Remove edge
                        neighbors_u_rem = np.where(adj[u_rem] == 1)[0]
                        neighbors_v_rem = np.where(adj[v_rem] == 1)[0]
                        
                        adj[u_rem, v_rem] = 0
                        adj[v_rem, u_rem] = 0
                        
                        for w in neighbors_u_rem:
                            if w != v_rem:
                                common[w, v_rem] -= 1
                                common[v_rem, w] -= 1
                        for w in neighbors_v_rem:
                            if w != u_rem:
                                common[u_rem, w] -= 1
                                common[w, u_rem] -= 1
                                
                        current_edge_count -= 1
                        
                        # Try to add 2 edges
                        added_count = 0
                        non_edges_swap = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                        rng.shuffle(non_edges_swap)
                        
                        for u_add, v_add in non_edges_swap:
                            if added_count == 2:
                                break
                            
                            neighbors_u_add = np.where(adj[u_add] == 1)[0]
                            neighbors_v_add = np.where(adj[v_add] == 1)[0]
                            
                            valid = True
                            if len(neighbors_u_add) > 0:
                                if np.max(common[neighbors_u_add, v_add]) > 1:
                                    valid = False
                            if valid and len(neighbors_v_add) > 0:
                                if np.max(common[u_add, neighbors_v_add]) > 1:
                                    valid = False
                            
                            if valid:
                                adj[u_add, v_add] = 1
                                adj[v_add, u_add] = 1
                                
                                for w in neighbors_u_add:
                                    if w != v_add:
                                        common[w, v_add] += 1
                                        common[v_add, w] += 1
                                for w in neighbors_v_add:
                                    if w != u_add:
                                        common[u_add, w] += 1
                                        common[w, u_add] += 1
                                
                                current_edge_count += 1
                                added_count += 1
                        
                        # Accept if we gained an edge
                        if current_edge_count > count_save:
                            improved = True
                            break
                        else:
                            # Revert
                            adj = adj_save
                            common = common_save
                            current_edge_count = count_save

        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    # Run multiple greedy builds with different seeds
    for trial in range(50):
        if time.time() > deadline - 15.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        edges = numpy_greedy_build(trial_rng)
        edges = verify(edges)
        
        if edges is not None:
            improved = numpy_local_search(edges, max_iters=200)
            improved = verify(improved)
            
            if improved is not None and len(improved) > best_e:
                best_e = len(improved)
                best = improved
            elif edges is not None and len(edges) > best_e:
                best_e = len(edges)
                best = edges
    
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
