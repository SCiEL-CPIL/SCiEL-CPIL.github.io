import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 145s to be safe.
    deadline = start + 145.0
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
        # Check K2,3-free: no two vertices have > 2 common neighbors
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
            
            # Check if adding (u,v) violates K2,3
            # For any w in N(u), common[w, v] must be <= 1 (so that after adding, it becomes <= 2? 
            # Wait. common[w, v] is the number of common neighbors of w and v BEFORE adding (u,v).
            # If we add (u,v), u becomes a common neighbor of w and v (if w!=u).
            # Also v becomes a common neighbor of u and w.
            # The constraint is: for any pair (x, y), |N(x) \cap N(y)| <= 2.
            # If we add (u, v):
            # 1. For any w in N(u) (w != v): u and w share v as a neighbor? No, v is now neighbor of u and w? 
            #    If w is in N(u), and we add (u,v), does v become neighbor of w? No, unless (w,v) exists.
            #    Let's re-evaluate the update logic.
            
            # Current common[w, v] = |N(w) \cap N(v)|.
            # If we add (u, v):
            # - v becomes a neighbor of u.
            # - u becomes a neighbor of v.
            
            # Pairs affected:
            # 1. Pair (u, w) where w in N(u):
            #    New common neighbor: v (since v is now in N(u) and v is already in N(w)? No, v is in N(w) only if (w,v) is an edge).
            #    Actually, the condition for validity is:
            #    For all w in N(u):
            #       The new common neighbors of (u, w) will include v IF v is in N(w).
            #       Wait, common[u, w] is |N(u) \cap N(w)|.
            #       If we add (u,v), v is added to N(u).
            #       So v is a new common neighbor of (u, w) if and only if v is in N(w).
            #       So we need: for all w in N(u), if (w,v) is an edge, then common[u, w] must be < 2 (so it can become 2).
            #       Actually, simpler: 
            #       The new common count for (u, w) is old_common[u, w] + (1 if v in N(w) else 0).
            #       This must be <= 2.
            #       So: if v in N(w), we require common[u, w] <= 1.
            #       If v not in N(w), common[u, w] remains same, so it's already <= 2.
            
            # 2. Pair (v, w) where w in N(v):
            #    Similarly, if u in N(w), we require common[v, w] <= 1.
            
            # 3. Pair (u, v) itself:
            #    New common count is |N(u) \cap N(v)|. This must be <= 2.
            #    Note: N(u) and N(v) do not include each other yet.
            #    So we just check if len(N(u) \cap N(v)) <= 2.

            # Let's check the intersection first
            if len(neighbors_u) > 0 and len(neighbors_v) > 0:
                # Check if u and v have > 2 common neighbors
                # neighbors_u are indices where adj[u]==1
                # neighbors_v are indices where adj[v]==1
                # Intersection:
                common_uv = np.sum(adj[u][neighbors_v]) # Count of neighbors of u that are also neighbors of v
                if common_uv > 2:
                    continue
            
            # Check constraints for neighbors of u
            valid = True
            if len(neighbors_u) > 0:
                # For each w in neighbors_u, if (w,v) is an edge, then common[u,w] must be <= 1
                # (w,v) is an edge means adj[w,v] == 1
                # We need common[u,w] <= 1 for all w in neighbors_u such that adj[w,v]==1
                # common[u, neighbors_u] gives the common counts for pairs (u, w)
                # We only care about w where adj[w,v]==1
                # Let's get the common counts for (u, w) for all w in neighbors_u
                common_u_w = common[u, neighbors_u]
                # Get the adjacency of v to these w's
                adj_v_to_w = adj[v, neighbors_u]
                # If adj_v_to_w[i] is 1, we need common_u_w[i] <= 1
                if np.any(adj_v_to_w == 1):
                    if np.max(common_u_w[adj_v_to_w == 1]) > 1:
                        valid = False
            
            if valid and len(neighbors_v) > 0:
                # Symmetric check for neighbors of v
                common_v_w = common[v, neighbors_v]
                adj_u_to_w = adj[u, neighbors_v]
                if np.any(adj_u_to_w == 1):
                    if np.max(common_v_w[adj_u_to_w == 1]) > 1:
                        valid = False
            
            if not valid:
                continue
            
            # Add edge
            adj[u, v] = 1
            adj[v, u] = 1
            
            # Update common counts
            # For w in neighbors_u:
            #   common[u, w] increases by 1 if v is in N(w) i.e., adj[w,v]==1
            #   common[w, u] increases by 1 if v is in N(w)
            #   Also, v is now a neighbor of u. So for any x in N(v), common[u, x] might change?
            #   No, the standard update is:
            #   Adding (u,v) increases common[x,y] by 1 if x is in N(u) and y is in N(v) ... wait.
            
            # Let's stick to the previous update logic which was verified to work, 
            # but ensure it matches the check logic.
            # Previous logic:
            # for w in neighbors_u:
            #    if w != v:
            #       common[w, v] += 1  # w and v now share u? No.
            #       common[v, w] += 1
            # This logic in the original code seemed to assume that adding (u,v) 
            # makes u a common neighbor of w and v? 
            # If w is in N(u), and we add (u,v), then u is in N(w) and u is in N(v)? 
            # No, u is in N(v) because we added (u,v). 
            # Is u in N(w)? Yes, because w is in N(u) and graph is undirected.
            # So u IS a common neighbor of w and v.
            # So common[w, v] increases by 1.
            # This holds for all w in N(u).
            
            # Similarly, v is in N(u) and v is in N(x) for all x in N(v).
            # So v is a common neighbor of u and x.
            # So common[u, x] increases by 1 for all x in N(v).
            
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

    def can_add_edge(adj, common, u, v):
        """Check if edge (u,v) can be added without violating K2,3-free constraint."""
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_v = np.where(adj[v] == 1)[0]
        
        # Check direct intersection
        if len(neighbors_u) > 0 and len(neighbors_v) > 0:
            common_uv = np.sum(adj[u][neighbors_v])
            if common_uv > 2:
                return False
        
        if len(neighbors_u) > 0:
            common_u_w = common[u, neighbors_u]
            adj_v_to_w = adj[v, neighbors_u]
            if np.any(adj_v_to_w == 1):
                if np.max(common_u_w[adj_v_to_w == 1]) > 1:
                    return False
            
        if len(neighbors_v) > 0:
            common_v_w = common[v, neighbors_v]
            adj_u_to_w = adj[u, neighbors_v]
            if np.any(adj_u_to_w == 1):
                if np.max(common_v_w[adj_u_to_w == 1]) > 1:
                    return False
        return True

    def add_edge(adj, common, u, v):
        """Add edge (u,v) and update common neighbor counts."""
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_v = np.where(adj[v] == 1)[0]
        
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

    def remove_edge(adj, common, u, v):
        """Remove edge (u,v) and update common neighbor counts."""
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_v = np.where(adj[v] == 1)[0]
        
        adj[u, v] = 0
        adj[v, u] = 0
        
        for w in neighbors_u:
            if w != v:
                common[w, v] -= 1
                common[v, w] -= 1
        for w in neighbors_v:
            if w != u:
                common[u, w] -= 1
                common[w, u] -= 1

    def local_search(edges, max_iters=100):
        """Enhanced local search with 1-for-2 and 2-for-3 swaps."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        current_edge_count = len(edges)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 15.0:
            improved = False
            iters += 1
            
            # Phase 1: Try adding any valid non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            rng.shuffle(non_edges)
            
            for u, v in non_edges:
                if can_add_edge(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    current_edge_count += 1
                    improved = True
                    break
            
            # Phase 2: Try 1-for-2 swaps
            if not improved and current_edge_count > 10:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                rng.shuffle(edge_list)
                
                for rem_u, rem_v in edge_list[:30]:
                    adj_save = adj.copy()
                    common_save = common.copy()
                    
                    remove_edge(adj, common, rem_u, rem_v)
                    
                    added = 0
                    for u, v in non_edges:
                        if adj[u, v]:
                            continue
                        if u == rem_u and v == rem_v:
                            continue
                        
                        if can_add_edge(adj, common, u, v):
                            add_edge(adj, common, u, v)
                            added += 1
                            if added == 2:
                                break
                    
                    if added >= 1:
                        current_edge_count -= 1 + (-added) # count = count - 1 + added
                        current_edge_count = current_edge_count - 1 + added
                        improved = True
                        break
                    else:
                        adj[:] = adj_save
                        common[:] = common_save

            # Phase 3: Try 2-for-3 swaps (Remove 2, Add 3)
            if not improved and current_edge_count > 15:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                rng.shuffle(edge_list)
                
                # Pick two edges to remove
                for i in range(min(10, len(edge_list))):
                    rem1 = edge_list[i]
                    # Pick a second edge different from the first
                    rem2_candidates = [e for j, e in enumerate(edge_list) if j != i]
                    if not rem2_candidates: continue
                    rem2 = rng.choice(rem2_candidates)
                    
                    adj_save = adj.copy()
                    common_save = common.copy()
                    
                    remove_edge(adj, common, rem1[0], rem1[1])
                    remove_edge(adj, common, rem2[0], rem2[1])
                    
                    added = 0
                    for u, v in non_edges:
                        if adj[u, v]:
                            continue
                        if (u == rem1[0] and v == rem1[1]) or (u == rem2[0] and v == rem2[1]):
                            continue
                        
                        if can_add_edge(adj, common, u, v):
                            add_edge(adj, common, u, v)
                            added += 1
                            if added == 3:
                                break
                    
                    if added >= 2: # Net gain >= 0, but we want net gain > 0 ideally. 
                                  # If added == 2, net 0. If added == 3, net +1.
                                  # Let's accept net 0 if it helps escape, but prefer net +1.
                                  # For simplicity, accept if added >= 2.
                        current_edge_count = current_edge_count - 2 + added
                        improved = True
                        break
                    else:
                        adj[:] = adj_save
                        common[:] = common_save
                    if improved: break
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    def final_polish(edges):
        """High-precision polish: try all possible single-edge additions and 2-edge swaps."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        current_edge_count = len(edges)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try all non-edges
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        continue
                    if can_add_edge(adj, common, u, v):
                        add_edge(adj, common, u, v)
                        current_edge_count += 1
                        improved = True
                        break
                if improved:
                    break
            
            # Try all 1-for-2 swaps
            if not improved:
                for rem_u in range(n):
                    for rem_v in range(rem_u + 1, n):
                        if not adj[rem_u, rem_v]:
                            continue
                        
                        adj_save = adj.copy()
                        common_save = common.copy()
                        
                        remove_edge(adj, common, rem_u, rem_v)
                        
                        added = 0
                        for u in range(n):
                            for v in range(u + 1, n):
                                if adj[u, v]:
                                    continue
                                if u == rem_u and v == rem_v:
                                    continue
                                
                                if can_add_edge(adj, common, u, v):
                                    add_edge(adj, common, u, v)
                                    added += 1
                                    if added == 2:
                                        break
                            if added == 2:
                                break
                        
                        if added == 2:
                            current_edge_count += 1
                            improved = True
                            break
                        else:
                            adj[:] = adj_save
                            common[:] = common_save
                    if improved:
                        break
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    # Run multiple greedy builds with different seeds
    for trial in range(30):
        if time.time() > deadline - 20.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        edges = numpy_greedy_build(trial_rng)
        edges = verify(edges)
        
        if edges is not None:
            # Local search with swaps
            improved = local_search(edges, max_iters=50)
            improved = verify(improved)
            
            if improved is not None:
                polished = final_polish(improved)
                polished = verify(polished)
                
                if polished is not None and len(polished) > best_e:
                    best_e = len(polished)
                    best = polished

    # Final verification
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
