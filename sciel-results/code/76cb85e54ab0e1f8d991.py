import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    n = 20
    
    # Precompute all triples for violation checking
    triples = list(combinations(range(n), 3))
    
    # Helper to get neighbors from mask
    def get_neighbors(adj, v):
        mask = adj[v]
        nbrs = []
        for x in range(n):
            if x != v and (mask >> x) & 1:
                nbrs.append(x)
        return nbrs

    def can_add(adj, u, v):
        """
        Fast check if adding edge (u, v) creates a K3,3.
        
        A K3,3 violation involving new edge (u,v) implies:
        1. u and v are in opposite partitions of the K3,3.
        2. Let Part A = {u, a, b} and Part B = {v, c, d}.
           Then v must be a common neighbor of a and b (v in N(a) and N(b)).
           And u must have at least 2 common neighbors with a and b 
           (excluding v, since u-v is the new edge).
           i.e., |N(u) ∩ N(a) ∩ N(b)| >= 2.
        
        Symmetric case: Part A = {v, a, b}, Part B = {u, c, d}.
           Then u in N(a) ∩ N(b) and |N(v) ∩ N(a) ∩ N(b)| >= 2.
        """
        # Case 1: v is in Part B. So v is a common neighbor of some a, b in Part A.
        # a, b must be neighbors of v.
        nbrs_v = get_neighbors(adj, v)
        for i in range(len(nbrs_v)):
            a = nbrs_v[i]
            if a == u: continue
            for j in range(i + 1, len(nbrs_v)):
                b = nbrs_v[j]
                if b == u: continue
                # Check if u has >= 2 common neighbors with a and b
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() >= 2:
                    return False
        
        # Case 2: u is in Part B. So u is a common neighbor of some a, b in Part A.
        nbrs_u = get_neighbors(adj, u)
        for i in range(len(nbrs_u)):
            a = nbrs_u[i]
            if a == v: continue
            for j in range(i + 1, len(nbrs_u)):
                b = nbrs_u[j]
                if b == v: continue
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() >= 2:
                    return False
        
        return True

    def find_violation(adj):
        """Find a K3,3 violation. Returns (u,v,w, common_mask) or None."""
        for u, v, w in triples:
            common = adj[u] & adj[v] & adj[w]
            if common.bit_count() >= 3:
                return (u, v, w, common)
        return None

    def repair(adj, rng):
        """Remove edges to fix violations."""
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            u, v, w, common = viol
            common_list = [x for x in range(n) if (common >> x) & 1]
            if not common_list:
                break
            
            # Try to remove an edge incident to one of the common neighbors
            # that breaks the violation
            removed = False
            rng.shuffle(common_list)
            for c in common_list:
                # Try removing (u,c), (v,c), or (w,c)
                for a in (u, v, w):
                    if (adj[a] >> c) & 1:
                        # Tentatively remove
                        adj[a] &= ~(1 << c)
                        adj[c] &= ~(1 << a)
                        
                        # Check if violation is gone
                        if find_violation(adj) is None:
                            removed = True
                            break
                        else:
                            # Revert
                            adj[a] |= (1 << c)
                            adj[c] |= (1 << a)
                if removed:
                    break
            if not removed:
                # Fallback: remove a random edge from the violation
                if common_list:
                    c = common_list[0]
                    a = u
                    if (adj[a] >> c) & 1:
                        adj[a] &= ~(1 << c)
                        adj[c] &= ~(1 << a)
                    else:
                        # Should not happen if common is correct, but safe fallback
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

    def greedy_add(adj, rng):
        """Add all possible edges greedily."""
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not ((adj[i] >> j) & 1)]
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    def local_search(adj, rng, max_iters=200):
        """
        Local search with 1-for-1 and 2-for-3 swaps.
        """
        adj = list(adj)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # 1. Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not ((adj[i] >> j) & 1)]
            if missing:
                rng.shuffle(missing)
                added = False
                for u, v in missing:
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        # Restart greedy after adding one to keep it simple/fast
                        break
                if added:
                    greedy_add(adj, rng)
                    continue
            
            # 2. If no edges can be added, try swaps
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
            if not edges:
                break
            
            # Try 2-for-3 swap (remove 2, add 3)
            success = False
            if len(edges) >= 2:
                # Sample a few 2-edge removals
                for _ in range(20):
                    if time.time() > deadline - 3:
                        break
                    e1_idx, e2_idx = rng.sample(range(len(edges)), 2)
                    e1 = edges[e1_idx]
                    e2 = edges[e2_idx]
                    
                    adj_temp = list(adj)
                    adj_temp[e1[0]] &= ~(1 << e1[1])
                    adj_temp[e1[1]] &= ~(1 << e1[0])
                    adj_temp[e2[0]] &= ~(1 << e2[1])
                    adj_temp[e2[1]] &= ~(1 << e2[0])
                    
                    # Try to add 3 edges
                    missing_temp = [(i, j) for i in range(n) for j in range(i + 1, n) 
                                    if not ((adj_temp[i] >> j) & 1)]
                    rng.shuffle(missing_temp)
                    added_count = 0
                    for u, v in missing_temp:
                        if added_count >= 3:
                            break
                        if can_add(adj_temp, u, v):
                            adj_temp[u] |= (1 << v)
                            adj_temp[v] |= (1 << u)
                            added_count += 1
                    
                    if added_count >= 3:
                        adj = adj_temp
                        greedy_add(adj, rng)
                        success = True
                        break
            
            if success:
                continue
            
            # Try 1-for-1 swap
            success = False
            if len(edges) >= 1:
                for _ in range(20):
                    if time.time() > deadline - 3:
                        break
                    r_idx = rng.randint(0, len(edges) - 1)
                    r_e = edges[r_idx]
                    
                    missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                               if not ((adj[i] >> j) & 1)]
                    if not missing:
                        break
                    a_idx = rng.randint(0, len(missing) - 1)
                    a_e = missing[a_idx]
                    
                    if r_e == a_e:
                        continue
                        
                    adj_temp = list(adj)
                    adj_temp[r_e[0]] &= ~(1 << r_e[1])
                    adj_temp[r_e[1]] &= ~(1 << r_e[0])
                    
                    if can_add(adj_temp, a_e[0], a_e[1]):
                        adj_temp[a_e[0]] |= (1 << a_e[1])
                        adj_temp[a_e[1]] |= (1 << a_e[0])
                        adj = adj_temp
                        greedy_add(adj, rng)
                        success = True
                        break
            
            if success:
                continue
            
            # Random kick: remove a random edge
            r_idx = rng.randint(0, len(edges) - 1)
            r_e = edges[r_idx]
            adj[r_e[0]] &= ~(1 << r_e[1])
            adj[r_e[1]] &= ~(1 << r_e[0])
            greedy_add(adj, rng)

    def edge_count(adj):
        return sum(m.bit_count() for m in adj) // 2

    def mask_to_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    best_adj = None
    best_edges = 0
    rng = random.Random(42)

    # Initialization 1: High density random
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        density = 0.5 + rng.random() * 0.3 # Start dense
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        repair(adj, rng)
        greedy_add(adj, rng)
        
        adj = local_search(adj, rng, max_iters=100)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # Early stop if we hit a very high count
        if best_edges > 180:
            break

    # Initialization 2: Structured (Circulant)
    if time.time() < deadline - 10:
        adj = [0] * n
        # Try circulant with steps 1, 2, 3, 4
        for d in range(1, 5):
            for i in range(n):
                j = (i + d) % n
                if i < j:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        repair(adj, rng)
        greedy_add(adj, rng)
        adj = local_search(adj, rng, max_iters=200)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)

    # Final polish
    if best_adj is not None and time.time() < deadline - 2:
        adj = list(best_adj)
        # A few more iterations of local search
        local_search(adj, rng, max_iters=50)
        greedy_add(adj, rng)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)

    if best_adj is None:
        return []

    return mask_to_edges(best_adj)
