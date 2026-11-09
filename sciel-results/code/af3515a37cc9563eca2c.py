import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Base: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    base_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    base_n = q
    
    # Function to check if a graph is valid (no K5, no independent 5)
    # Represent graph as list of bitmasks for adjacency
    def check_valid(n, adj_masks):
        # Check for K5
        for i in range(n):
            # Get neighbors of i
            # We need to find a K4 in the neighborhood of i
            # To speed up, iterate over pairs of neighbors?
            # Simpler: iterate over all 5-combinations is too slow in Python loops.
            # Let's use a recursive backtracking or a specific K5 checker.
            pass
        return False

    # A faster checker for small n:
    # Precompute all 5-combinations? No.
    # For each i, let N_i be the set of neighbors.
    # A K5 exists iff there is an i and a K4 in N_i.
    # A K4 in a set S exists iff there are i,j,k,l in S such that all edges exist.
    # For n=38, average degree ~19. C(19,4) ~ 3876.
    # For each i, check K4 in N_i.
    # Total checks: n * C(deg, 4).
    
    def has_k5(n, adj_masks):
        for i in range(n):
            # Get list of neighbors
            neigh = []
            mask = adj_masks[i]
            # Extract bits
            # This is slow in Python. 
            # Better: use integer bit operations.
            # To find K4 in neighbors:
            # Iterate over pairs (j, k) in neighbors.
            # Check if there exist (l, m) in neighbors such that ...
            # Actually, checking K4 in a graph of size ~20 is fast.
            
            # Let's optimize: 
            # For each i, get neighbors list.
            # For each pair (j, k) in neighbors:
            #   common = neighbors_j & neighbors_k & neighbors_i (mask)
            #   Check if common contains a K2 (edge) among vertices > k?
            #   This is getting complicated.
            
            # Alternative: Use a simple O(n^5) check but with bitsets and early termination?
            # C(38, 5) = 501,942.
            # 500k iterations in Python is ~0.5s.
            # If we do this 100 times, it's 50s. Acceptable.
            
            pass
        return False

    # Let's implement a simple checker using combinations
    from itertools import combinations
    
    def is_valid_graph(n, edges_set, adj_masks):
        # adj_masks[i] is int with bits set for neighbors of i
        # Check K5
        for comb in combinations(range(n), 5):
            i, j, k, l, m = comb
            # Check if all edges exist
            if (adj_masks[i] >> j) & 1 and (adj_masks[i] >> k) & 1 and (adj_masks[i] >> l) & 1 and (adj_masks[i] >> m) & 1 and \
               (adj_masks[j] >> k) & 1 and (adj_masks[j] >> l) & 1 and (adj_masks[j] >> m) & 1 and \
               (adj_masks[k] >> l) & 1 and (adj_masks[k] >> m) & 1 and \
               (adj_masks[l] >> m) & 1:
                return False
            # Check if no edges exist (Independent 5)
            if not ((adj_masks[i] >> j) & 1) and not ((adj_masks[i] >> k) & 1) and not ((adj_masks[i] >> l) & 1) and not ((adj_masks[i] >> m) & 1) and \
               not ((adj_masks[j] >> k) & 1) and not ((adj_masks[j] >> l) & 1) and not ((adj_masks[j] >> m) & 1) and \
               not ((adj_masks[k] >> l) & 1) and not ((adj_masks[k] >> m) & 1) and \
               not ((adj_masks[l] >> m) & 1):
                return False
        return True

    # Local Search
    def local_search(target_n, initial_edges, time_limit):
        # Build adj masks
        adj_masks = [0] * target_n
        for u, v in initial_edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        # Precompute all 5-combinations for speed?
        # For n=38, C(38,5) is ~500k. Storing them is fine.
        combs = list(combinations(range(target_n), 5))
        
        def count_violations():
            viol = 0
            for comb in combs:
                i, j, k, l, m = comb
                # Check K5
                if (adj_masks[i] >> j) & 1 and (adj_masks[i] >> k) & 1 and (adj_masks[i] >> l) & 1 and (adj_masks[i] >> m) & 1 and \
                   (adj_masks[j] >> k) & 1 and (adj_masks[j] >> l) & 1 and (adj_masks[j] >> m) & 1 and \
                   (adj_masks[k] >> l) & 1 and (adj_masks[k] >> m) & 1 and \
                   (adj_masks[l] >> m) & 1:
                    viol += 1
                    continue
                # Check Ind5
                if not ((adj_masks[i] >> j) & 1) and not ((adj_masks[i] >> k) & 1) and not ((adj_masks[i] >> l) & 1) and not ((adj_masks[i] >> m) & 1) and \
                   not ((adj_masks[j] >> k) & 1) and not ((adj_masks[j] >> l) & 1) and not ((adj_masks[j] >> m) & 1) and \
                   not ((adj_masks[k] >> l) & 1) and not ((adj_masks[k] >> m) & 1) and \
                   not ((adj_masks[l] >> m) & 1):
                    viol += 1
            return viol

        current_viol = count_violations()
        best_viol = current_viol
        best_adj = adj_masks[:]
        
        while time.time() < time_limit:
            # Pick a random edge to flip
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            # Flip
            adj_masks[u] ^= (1 << v)
            adj_masks[v] ^= (1 << u)
            
            new_viol = count_violations()
            
            if new_viol < current_viol:
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = adj_masks[:]
                    if best_viol == 0:
                        return best_adj
            else:
                # Revert
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
            # Simple restart if stuck?
            # For now, just keep flipping.
            
        # Return best found
        return best_adj if best_viol == 0 else None

    # Try to extend to 38
    # Start with Paley 37, add vertex 37
    # Connect vertex 37 to random vertices?
    # Or better: Start with random graph of size 38?
    # Random graph has many violations.
    # Let's try to grow from Paley 37.
    
    # Strategy:
    # 1. Try local search on n=38 starting from Paley 37 + random vertex.
    # 2. If fails, try n=39.
    
    for target_n in [38, 39, 40]:
        if time.time() > deadline - 10:
            break
            
        # Initialize
        init_edges = base_edges[:]
        if target_n > base_n:
            # Add vertices base_n to target_n-1
            # Connect them randomly to existing vertices
            for v in range(base_n, target_n):
                for u in range(v):
                    if random.random() < 0.5:
                        init_edges.append((u, v))
            
        # Run local search
        time_slice = 30.0
        end_time = time.time() + time_slice
        result = local_search(target_n, init_edges, end_time)
        
        if result:
            # Convert adj_masks to edges
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if (result[i] >> j) & 1:
                        edges.append((i, j))
            return target_n, edges
            
    # Fallback to seed
    return base_n, base_edges
