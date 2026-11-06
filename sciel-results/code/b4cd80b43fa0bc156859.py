import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s wall-clock limit)

    def time_left():
        return deadline - time.time()

    def has_c5_fast(adj, n):
        """Check if graph has a 5-cycle using adjacency sets. Returns True if C5 found."""
        # For each pair of non-adjacent vertices (u,v), check if there's a path of length 3 between them
        # that doesn't use u or v internally, forming a C5.
        # Actually, a C5 is v0-v1-v2-v3-v4-v0. 
        # Alternative: for each edge (u,v), check if there's a path of length 3 from u to v not using the edge directly.
        # That would form a C4+edge or C5? No.
        
        # Better: iterate over all 5-tuples is too slow.
        # Use the fact that C5 exists iff there exist two non-adjacent vertices with a common neighbor at distance 2 and 3.
        
        # Fast check: For each vertex v, look at its neighbors N(v).
        # A C5 through v: v-a-b-c-d-v where a,d in N(v), b in N(a)∩N(c), c in N(d)∩N(b).
        # Simpler: for each pair of non-adjacent vertices (a,d), check if there exist b,c such that
        # a-b, b-c, c-d are edges and a,d not adjacent, and all 5 distinct.
        # This is still complex.
        
        # Most efficient for n<=35: iterate over all 5-combinations is C(35,5)=324,632 - feasible!
        # But checking each is O(1) with adjacency sets.
        
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 in (v0, v1):
                        continue
                    for v3 in adj[v2]:
                        if v3 in (v0, v1, v2):
                            continue
                        for v4 in adj[v3]:
                            if v4 in (v0, v1, v2, v3):
                                continue
                            if v0 in adj[v4]:
                                return True
        return False

    def has_ind_set_9_fast(adj, n):
        """Check if there's an independent set of size 9 using backtracking with pruning."""
        # Convert to adjacency bitmasks for speed
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask
        
        def search(candidates_mask, depth):
            if depth >= 9:
                return True
            # Count bits in candidates_mask
            if bin(candidates_mask).count('1') < depth:
                return False
            if candidates_mask == 0:
                return False
            
            # Pick a vertex with minimum degree in the induced subgraph
            # to maximize pruning
            best_v = -1
            best_deg = n + 1
            cm = candidates_mask
            while cm:
                lb = cm & (-cm)
                v = lb.bit_length() - 1
                # Count neighbors of v within candidates
                deg = bin(adj_mask[v] & candidates_mask).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                cm ^= lb
            
            if best_v == -1:
                return False
            
            v_mask = 1 << best_v
            
            # Branch 1: include best_v
            new_cands = candidates_mask & ~v_mask & ~adj_mask[best_v]
            if search(new_cands, depth + 1):
                return True
            
            # Branch 2: exclude best_v
            if search(candidates_mask & ~v_mask, depth):
                return True
            
            return False
        
        return search((1 << n) - 1, 0)

    def is_valid(n, adj):
        if has_c5_fast(adj, n):
            return False
        if has_ind_set_9_fast(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    # Known valid: 8 copies of K4 (n=32)
    def make_8k4():
        n = 32
        adj = [set() for _ in range(n)]
        for b in range(8):
            for i in range(4):
                for j in range(i+1, 4):
                    u = 4*b + i
                    v = 4*b + j
                    adj[u].add(v)
                    adj[v].add(u)
        return n, adj

    n_best, adj_best = make_8k4()
    
    # Verify the known solution
    if not is_valid(n_best, adj_best):
        # Fallback: empty graph
        n_best = 1
        adj_best = [set()]

    random.seed(42)
    
    # Try to find n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time_left() < 15:
            break
        
        found = False
        
        # Strategy 1: Random graphs with edge probability tuned
        for attempt in range(500):
            if time_left() < 5:
                break
            
            # For n=33: need alpha<=8, so complement has no K9.
            # Edge probability around 0.7-0.85
            p = random.uniform(0.65, 0.90)
            
            adj = [set() for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Quick C5 check
            if has_c5_fast(adj, target_n):
                continue
            
            # Check independent set
            if has_ind_set_9_fast(adj, target_n):
                continue
            
            # Valid!
            n_best = target_n
            adj_best = adj
            found = True
            break
        
        if found:
            break
        
        # Strategy 2: Start from 8xK4 and add vertices
        if target_n == 33 and time_left() > 20:
            for attempt in range(300):
                if time_left() < 5:
                    break
                
                base_n, base_adj = make_8k4()
                adj = [set(s) for s in base_adj]
                adj.append(set())
                n = 33
                
                # Connect vertex 32 to a random subset
                # To keep alpha < 9, vertex 32 must "hit" all independent sets of size 8 in the base
                # Base has alpha=8 (one from each K4)
                # So vertex 32 must be adjacent to at least one vertex in each K4 block? No.
                # An independent set of size 9 in the new graph would be:
                #   - Either includes vertex 32: then need 8 more independent vertices from base that are NOT adjacent to 32.
                #   - Or doesn't include vertex 32: then need 9 independent vertices from base. But base alpha=8, so impossible.
                # So we just need: the subgraph induced by non-neighbors of 32 in base has alpha <= 7.
                
                # Random connection
                for i in range(32):
                    if random.random() < 0.5:
                        adj[32].add(i)
                        adj[i].add(32)
                
                if has_c5_fast(adj, n):
                    continue
                
                if has_ind_set_9_fast(adj, n):
                    continue
                
                n_best = 33
                adj_best = adj
                found = True
                break
        
        if found:
            break
    
    return {'n': n_best, 'edges': adj_to_edges(adj_best, n_best)}
