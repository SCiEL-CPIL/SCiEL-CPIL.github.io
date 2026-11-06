import time
import random
import itertools
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 20
    best_edges = 18  # K2,18 baseline
    best_edge_list = [[a, b] for a in range(2) for b in range(2, 20)]
    
    def count_k33_violations(adj):
        """Count number of K3,3 violations (triples with >=3 common neighbors)."""
        violations = 0
        for triple in itertools.combinations(range(n), 3):
            u, v, w = triple
            # Common neighbors of u, v, w
            common = adj[u] & adj[v] & adj[w]
            if common.sum() >= 3:
                violations += 1
        return violations
    
    def is_valid(adj):
        return count_k33_violations(adj) == 0
    
    def edge_count(adj):
        return int(adj.sum() // 2)
    
    def try_remove_to_fix(adj, max_time=None):
        """Remove edges to eliminate all K3,3 violations. Returns (adj, removed_count)."""
        removed = 0
        while count_k33_violations(adj) > 0:
            if max_time is not None and time.time() > max_time:
                return adj, removed
            # Find a violating triple and remove an edge from it
            for triple in itertools.combinations(range(n), 3):
                u, v, w = triple
                common = adj[u] & adj[v] & adj[w]
                if common.sum() >= 3:
                    # Remove one edge: pick a common neighbor x, remove edge u-x
                    # Actually, we need to remove an edge that breaks this violation
                    # The violation is: {u,v,w} and {x1,x2,x3} form K3,3
                    # Remove edge u-x1 (or any edge between the two parts)
                    common_list = np.where(common)[0]
                    # Remove edge between u and first common neighbor
                    x = common_list[0]
                    adj[u, x] = 0
                    adj[x, u] = 0
                    removed += 1
                    break
            else:
                break
        return adj, removed
    
    def try_add_edges(adj, max_time=None):
        """Try to add edges one at a time to increase density while staying K3,3-free."""
        added = 0
        progress = True
        while progress:
            if max_time is not None and time.time() > max_time:
                break
            progress = False
            # Try all possible missing edges
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i, j]:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        if is_valid(adj):
                            added += 1
                            progress = True
                        else:
                            adj[i, j] = 0
                            adj[j, i] = 0
        return adj, added
    
    def swap_edges(adj, max_time=None):
        """Try 2-for-2 edge swaps: remove edges (a,b) and (c,d), add (a,c) and (b,d)."""
        swaps = 0
        # Collect current edges
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edge_set.add((i, j))
        
        # Try random swaps
        edges_list = list(edge_set)
        random.shuffle(edges_list)
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts:
            if max_time is not None and time.time() > max_time:
                break
            attempts += 1
            # Pick two edges to remove
            if len(edges_list) < 2:
                break
            e1 = edges_list[random.randint(0, len(edges_list)-1)]
            e2 = edges_list[random.randint(0, len(edges_list)-1)]
            if e1 == e2:
                continue
            a, b = e1
            c, d = e2
            # Try to add (a,c) and (b,d) or (a,d) and (b,c)
            for (x1, y1), (x2, y2) in [((a,c),(b,d)), ((a,d),(b,c))]:
                if x1 == y1 or x2 == y2:
                    continue
                if (x1, y1) in edge_set or (min(x1,y1), max(x1,y1)) in edge_set:
                    continue
                if (x2, y2) in edge_set or (min(x2,y2), max(x2,y2)) in edge_set:
                    continue
                # Remove e1, e2
                adj[a, b] = 0; adj[b, a] = 0
                adj[c, d] = 0; adj[d, c] = 0
                edge_set.discard(e1)
                edge_set.discard(e2)
                # Add new edges
                adj[x1, y1] = 1; adj[y1, x1] = 1
                adj[x2, y2] = 1; adj[y2, x2] = 1
                new_e1 = (min(x1,y1), max(x1,y1))
                new_e2 = (min(x2,y2), max(x2,y2))
                edge_set.add(new_e1)
                edge_set.add(new_e2)
                if is_valid(adj):
                    swaps += 1
                    # Update edges_list
                    edges_list = list(edge_set)
                    random.shuffle(edges_list)
                else:
                    # Revert
                    adj[x1, y1] = 0; adj[y1, x1] = 0
                    adj[x2, y2] = 0; adj[y2, x2] = 0
                    edge_set.discard(new_e1)
                    edge_set.discard(new_e2)
                    adj[a, b] = 1; adj[b, a] = 1
                    adj[c, d] = 1; adj[d, c] = 1
                    edge_set.add(e1)
                    edge_set.add(e2)
        return adj, swaps
    
    # Multiple random restarts with different densities
    for restart in range(30):
        if time.time() > deadline:
            break
        
        random.seed(42 + restart * 1000)
        
        # Start with random graph at various densities
        p = 0.5 + 0.3 * (restart % 5) / 5  # 0.5 to 0.74
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Fix violations
        adj, _ = try_remove_to_fix(adj, max_time=deadline)
        
        # Try to add edges
        adj, _ = try_add_edges(adj, max_time=deadline)
        
        # Try swaps to improve
        adj, _ = swap_edges(adj, max_time=deadline)
        
        # Another round of adding
        adj, _ = try_add_edges(adj, max_time=deadline)
        
        # Check validity
        if is_valid(adj):
            ec = edge_count(adj)
            if ec > best_edges:
                best_edges = ec
                best_edge_list = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j]]
    
    # Final check
    if best_edges <= 18:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edge_list
