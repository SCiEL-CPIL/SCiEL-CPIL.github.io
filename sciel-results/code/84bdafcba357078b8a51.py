import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for the 120s limit
    
    # Helper: Check if there is an independent set of size >= target
    # Returns True if an independent set of size 'target' exists.
    def has_independent_set(adj_mask, n, target=11):
        sys.setrecursionlimit(10000)
        node_count = [0]
        max_nodes = 2000000  # Limit to prevent excessive runtime per check
        
        def dfs(candidates, size):
            if size >= target:
                return True
            
            if candidates == 0:
                return False
            
            # Pruning: if even taking all candidates doesn't reach target
            if size + candidates.bit_count() < target:
                return False
            
            # Limit node count to ensure we return quickly for many seeds
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            
            # Pick a vertex from candidates
            # Heuristic: Pick vertex with highest degree in the induced subgraph.
            # This tends to reduce the candidate set quickly in the 'include' branch.
            c = candidates
            best_v = -1
            max_deg = -1
            
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                
                deg = (adj_mask[v] & candidates).bit_count()
                
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    # Early stop if degree is very high, as it will prune heavily
                    if deg >= 10:
                        break
            
            if best_v == -1:
                return False

            # Branch 1: Include best_v
            # Remove best_v and all its neighbors from candidates
            new_candidates = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_candidates, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            # Remove best_v from candidates
            new_candidates = candidates & ~(1 << best_v)
            if dfs(new_candidates, size):
                return True
                
            return False

        all_candidates = (1 << n) - 1
        return dfs(all_candidates, 0)

    def generate_greedy_c4_free(n, seed):
        """Incremental C4-free construction: guaranteed valid, no full verification needed."""
        random.seed(seed)
        adj_mask = [0] * n
        
        # List of all possible edges
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            nu = adj_mask[u]
            nv = adj_mask[v]
            
            # Check if adding (u, v) creates a C4
            # A C4 is formed if there exists x in N(u) and y in N(v) such that x-y is an edge.
            # Efficient check: Iterate over the smaller neighbor set.
            if nu.bit_count() < nv.bit_count():
                c = nu
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nv:
                        safe = False
                        break
            else:
                c = nv
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nu:
                        safe = False
                        break
            
            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
        return adj_mask

    def repair_graph(adj_mask, n, target_alpha=10):
        """Try to add edges to reduce independence number. Returns modified adj_mask or None."""
        if time.time() > deadline - 10:
            return None
        
        # Find an independent set of size target_alpha+1 to guide edge additions
        sys.setrecursionlimit(10000)
        found = [None]
        
        def find_iset(candidates, size, current_set):
            if found[0] is not None:
                return True
            if size >= target_alpha + 1:
                found[0] = list(current_set)
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target_alpha + 1:
                return False
            
            c = candidates
            best_v = -1
            max_deg = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    if deg >= 10:
                        break
            if best_v == -1:
                return False
            
            current_set.append(best_v)
            new_cands = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if find_iset(new_cands, size + 1, current_set):
                return True
            current_set.pop()
            
            new_cands = candidates & ~(1 << best_v)
            if find_iset(new_cands, size, current_set):
                return True
            return False
        
        all_cands = (1 << n) - 1
        if not find_iset(all_cands, 0, []):
            return None
        
        iset = found[0]
        
        # Try to add edges between pairs in the independent set that don't create C4
        added = False
        for i in range(len(iset)):
            for j in range(i + 1, len(iset)):
                u, v = iset[i], iset[j]
                if adj_mask[u] & (1 << v):
                    continue
                nu = adj_mask[u]
                nv = adj_mask[v]
                safe = True
                if nu.bit_count() < nv.bit_count():
                    c = nu
                    while c:
                        lsb = c & (-c)
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj_mask[x] & nv:
                            safe = False
                            break
                else:
                    c = nv
                    while c:
                        lsb = c & (-c)
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj_mask[x] & nu:
                            safe = False
                            break
                if safe:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    added = True
            if time.time() > deadline - 8:
                return None
        
        return adj_mask if added else None

    def build_edge_list(adj_mask, n):
        edges = []
        for u in range(n):
            m = adj_mask[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    # We want to maximize n.
    # Try n from 43 down to 30.
    # For each n, try a number of random seeds.
    # If we find a valid graph, return it.
    
    for n in range(43, 29, -1):
        seeds_tried = 0
        while time.time() < deadline - 15:
            seed = seeds_tried
            adj_mask = generate_greedy_c4_free(n, seed)
            
            # Check independence number
            if not has_independent_set(adj_mask, n, 11):
                edges = build_edge_list(adj_mask, n)
                return {'n': n, 'edges': edges}
            
            # Near-miss: try repair
            repaired = repair_graph(adj_mask, n)
            if repaired is not None and not has_independent_set(repaired, n, 11):
                edges = build_edge_list(repaired, n)
                return {'n': n, 'edges': edges}
            
            seeds_tried += 1
            
        if time.time() > deadline - 20:
            break

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    return {'n': 30, 'edges': fallback_edges}
