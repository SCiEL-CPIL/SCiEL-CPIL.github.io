import numpy as np
import random
import time
import sys

N = 31
E = 91
DEADLINE_SECONDS = 118.0  # safety margin under the 150 s limit

def get_points_q5():
    """
    Generate the 31 points of PG(2,5) as representative vectors in F_5^3.
    """
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    return points

def build_polarity_graph(M):
    """
    Construct the polarity graph of PG(2,5) using matrix M.
    Returns adjacency list (bitmasks) and edge list.
    """
    points = get_points_q5()
    adj_masks = [0] * N
    edges = []
    
    for i in range(N):
        for j in range(i + 1, N):
            pi = np.array(points[i])
            pj = np.array(points[j])
            # Check if pi is on polar line of pj: pi^T M pj = 0
            val = (pi @ M @ pj) % 5
            if val == 0:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                edges.append((i, j))
                
    return adj_masks, edges

def count_c4_bitmask(adj_masks):
    """
    Count C4s using bitmasks.
    C4 = 1/2 * sum_{u < v} C(popcount(adj[u] & adj[v]), 2)
    """
    total = 0
    for u in range(N):
        for v in range(u + 1, N):
            common = adj_masks[u] & adj_masks[v]
            k = bin(common).count('1')
            total += k * (k - 1) // 2
    return total // 2

def count_new_c4s_for_add(adj_masks, u, v):
    """
    Count the number of new C4s created by adding edge (u, v) to the graph represented by adj_masks.
    This is the number of edges between N(u) and N(v) in the current graph.
    """
    # N(u) and N(v) do not include u or v themselves in the bitmask representation 
    # if we assume no self-loops, which is true here.
    # However, we must ensure we don't count the edge (u,v) itself if it were present, 
    # but it's not.
    # We need to count edges (w, x) such that w in N(u), x in N(v), and (w,x) is an edge.
    # Since u and v are not neighbors, N(u) and N(v) are disjoint? No, they can overlap.
    # But w cannot be v, and x cannot be u.
    # In the bitmask, bit v is not set in adj_masks[u], and bit u is not set in adj_masks[v].
    
    nu = adj_masks[u]
    nv = adj_masks[v]
    
    count = 0
    # Iterate over neighbors of u
    w = nu
    while w:
        lsb = w & -w
        wi = lsb.bit_length() - 1
        w ^= lsb
        # Check if wi is connected to any neighbor of v
        # common = nv & adj_masks[wi]
        # But we must ensure wi != v (true, as (u,v) is not an edge)
        # And we must ensure the other end x != u (true, as (u,v) is not an edge)
        # Also, we must avoid double counting? 
        # The formula "number of edges between N(u) and N(v)" counts each edge once if we just sum.
        # An edge (w, x) with w in N(u) and x in N(v) is counted when we pick w.
        # Is it possible that x is also in N(u)? Yes.
        # If x is in N(u) and N(v), then (w, x) is an edge.
        # This creates a C4: u-w-x-v-u.
        # This is a valid C4.
        # So we just count all such edges.
        common = nv & adj_masks[wi]
        count += bin(common).count('1')
        
    return count

def construct():
    start_time = time.time()
    
    # 1. Try multiple polarity forms
    # Standard forms for PG(2, q) with q odd
    # We want a symmetric matrix M such that the graph is C4-free and has ~90 edges.
    # For q=5, the number of points is 31.
    # Expected edges for polarity graph: q(q+1)/2 = 5*6/2 = 15? No.
    # In PG(2, q), the number of absolute points is q+1.
    # The number of edges is (q(q+1)/2) + (q+1) ? No.
    # Let's just test a few matrices.
    
    forms = [
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int), # Identity
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
    ]
    
    best_initial_graph = None
    best_initial_c4 = float('inf')
    
    for M in forms:
        if time.time() > start_time + 10: break
        
        adj_masks, edges = build_polarity_graph(M)
        
        # We need 91 edges. Polarity graphs often have specific counts.
        # If we have 90, we add 1. If 91, we are good. If 89, we add 2.
        # Ideally, we find a 90-edge C4-free graph.
        
        # Adjust edge count to 91
        current_edges = edges[:]
        current_edge_set = set(current_edges)
        
        # Rebuild masks if we need to add/remove
        # For simplicity, let's assume we start with the generated graph and adjust.
        
        # If len < 91, add non-edges greedily
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        non_edges = [e for e in all_edges if e not in current_edge_set]
        
        # Sort non-edges by cost (new C4s)
        # This is expensive if done for all, but N is small.
        
        temp_masks = adj_masks[:]
        
        # Ensure we have 91 edges
        while len(current_edges) < E:
            if not non_edges: break
            best_idx = 0
            best_cost = float('inf')
            # To save time, only check a subset if non_edges is large, but 465 is small.
            for idx, e in enumerate(non_edges):
                u, v = e
                cost = count_new_c4s_for_add(temp_masks, u, v)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            
            u, v = non_edges[best_idx]
            non_edges.pop(best_idx)
            current_edges.append((u, v))
            current_edge_set.add((u, v))
            temp_masks[u] |= (1 << v)
            temp_masks[v] |= (1 << u)
            
        while len(current_edges) > E:
            # Remove random edge (or the one contributing most C4s)
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            current_edges.pop(idx)
            current_edge_set.remove((u, v))
            temp_masks[u] &= ~(1 << v)
            temp_masks[v] &= ~(1 << u)
            
        c4_count = count_c4_bitmask(temp_masks)
        
        if c4_count < best_initial_c4:
            best_initial_c4 = c4_count
            best_initial_graph = {
                'edges': current_edges,
                'masks': temp_masks
            }
            
    if best_initial_graph is None:
        # Fallback: random graph
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        edges = all_edges[:E]
        masks = [0]*N
        for u,v in edges:
            masks[u] |= 1<<v
            masks[v] |= 1<<u
        best_initial_graph = {'edges': edges, 'masks': masks}
        best_initial_c4 = count_c4_bitmask(masks)
        
    current_edges = best_initial_graph['edges'][:]
    current_masks = best_initial_graph['masks'][:]
    current_c4 = best_initial_c4
    
    best_edges = current_edges[:]
    best_c4 = current_c4
    
    # 2. Local Search (Simulated Annealing)
    T = 5.0
    cooling = 0.99
    deadline = start_time + DEADLINE_SECONDS
    
    # Precompute edge set for fast lookup
    current_edge_set = set(current_edges)
    
    while time.time() < deadline - 2:
        # Pick a random edge to remove
        remove_idx = random.randint(0, E - 1)
        u1, v1 = current_edges[remove_idx]
        
        # Pick a random non-edge to add
        valid = False
        u2, v2 = 0, 0
        for _ in range(50): # Try more times to find a non-edge
            u2 = random.randint(0, N - 1)
            v2 = random.randint(0, N - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) not in current_edge_set:
                valid = True
                break
        
        if not valid:
            continue
            
        # Calculate Delta C4
        # 1. Remove (u1, v1)
        #    Decrease = C4s containing (u1, v1)
        #    C4s containing (u1, v1) = edges between N(u1)\{v1} and N(v1)\{u1}
        
        # Temporarily remove (u1, v1) from masks to calculate the "add" part correctly?
        # Or calculate add part on G \ {(u1,v1)} and remove part on G.
        
        # Let's do it carefully.
        # G is current graph.
        # G' = G \ {(u1,v1)} + {(u2,v2)}
        
        # Delta = C4(G') - C4(G)
        # C4(G') = C4(G) - C4_containing(u1,v1) + C4_new_from(u2,v2 in G')
        
        # C4_containing(u1,v1) in G:
        # Edges between N_G(u1) \ {v1} and N_G(v1) \ {u1}
        nu1 = current_masks[u1] & ~(1 << v1)
        nv1 = current_masks[v1] & ~(1 << u1)
        
        remove_count = 0
        w = nu1
        while w:
            lsb = w & -w
            wi = lsb.bit_length() - 1
            w ^= lsb
            # Count edges from wi to any neighbor in nv1
            common = nv1 & current_masks[wi]
            remove_count += bin(common).count('1')
        
        # Now, calculate C4_new_from(u2,v2) in G' = G \ {(u1,v1)}
        # We need N_{G'}(u2) and N_{G'}(v2)
        
        nu2 = current_masks[u2]
        nv2 = current_masks[v2]
        
        # If u2 or v2 is u1 or v1, we need to adjust masks
        # Case 1: (u2,v2) is disjoint from (u1,v1)
        if u2 not in (u1, v1) and v2 not in (u1, v1):
            # Masks are same as G
            pass
        else:
            # Adjust nu2, nv2 to reflect removal of (u1,v1)
            if u1 == u2:
                nu2 &= ~(1 << v1)
            elif v1 == u2:
                nu2 &= ~(1 << u1)
            
            if u1 == v2:
                nv2 &= ~(1 << v1)
            elif v1 == v2:
                nv2 &= ~(1 << u1)
                
            # Also, if u2 is neighbor of u1, and u1 is removed... 
            # Wait, nu2 is the set of neighbors of u2 in G.
            # In G', we removed edge (u1, v1).
            # This only affects nu2 if u1 is a neighbor of u2 (i.e., u1 in nu2).
            # If u1 in nu2, then in G', is u1 still a neighbor of u2?
            # Yes, unless the edge (u1, u2) was removed. But we only removed (u1, v1).
            # So if u1 != v1, the edge (u1, u2) is still there.
            # The only change is that u1 and v1 are no longer connected to each other.
            # So for any vertex x, N_{G'}(x) = N_G(x) \ {y if (x,y) was (u1,v1) or (v1,u1)}
            # So if u2 is not u1 or v1, N_{G'}(u2) = N_G(u2).
            # If u2 == u1, N_{G'}(u1) = N_G(u1) \ {v1}.
            # If u2 == v1, N_{G'}(v1) = N_G(v1) \ {u1}.
            
            # My logic above handles this:
            # if u2 == u1: remove v1 from nu2.
            # if u2 == v1: remove u1 from nu2.
            # Same for nv2.
            
            # Wait, what if u2 == u1 and v2 == v1? Then we are adding back the same edge?
            # We ensured (u2,v2) is a non-edge, and (u1,v1) is an edge. So they are different.
            
            # What if u2 == u1 and v2 == u1? Impossible, u2 != v2.
            # What if u2 == u1 and v2 == v1? Impossible.
            # What if u2 == v1 and v2 == u1? Then (u2,v2) = (v1,u1) which is same as (u1,v1).
            # But (u1,v1) is an edge, so (u2,v2) would be an edge.
            # But we picked (u2,v2) as a non-edge. So this case is impossible.
            
            # So the adjustments are correct.
        
        # Now count edges between nu2 and nv2 in G'
        # Note: nu2 and nv2 are bitmasks representing neighbors in G'.
        add_count = 0
        w = nu2
        while w:
            lsb = w & -w
            wi = lsb.bit_length() - 1
            w ^= lsb
            # We need to count edges (wi, x) where x in nv2.
            # But we must use the adjacency in G'.
            # If wi is u1 or v1, its adjacency changed.
            # Specifically, if wi == u1, N_{G'}(u1) = nu2 (if u2==u1) or current_masks[u1] \ {v1}.
            # This is getting complicated.
            
            # Alternative: Just update the masks, count, then revert.
            # It's simpler and less error-prone for small N.
            
            # Let's abort this delta calculation and do the simple update/revert.
            # N=31, counting C4s is fast.
            break
        
        # Simple Update/Revert Strategy
        # Remove (u1, v1)
        current_masks[u1] &= ~(1 << v1)
        current_masks[v1] &= ~(1 << u1)
        current_edge_set.remove((u1, v1))
        
        # Add (u2, v2)
        current_masks[u2] |= (1 << v2)
        current_masks[v2] |= (1 << u2)
        current_edge_set.add((u2, v2))
        current_edges[remove_idx] = (u2, v2)
        
        new_c4 = count_c4_bitmask(current_masks)
        delta = new_c4 - current_c4
        
        if delta <= 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
            current_c4 = new_c4
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = current_edges[:]
            T *= cooling
        else:
            # Revert
            current_masks[u2] &= ~(1 << v2)
            current_masks[v2] &= ~(1 << u2)
            current_edge_set.remove((u2, v2))
            
            current_masks[u1] |= (1 << v1)
            current_masks[v1] |= (1 << u1)
            current_edge_set.add((u1, v1))
            
            current_edges[remove_idx] = (u1, v1)
            T *= cooling
            
        if best_c4 <= 1:
            break
            
    return best_edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges.")
    # Verify count
    masks = [0]*N
    for u,v in edges:
        masks[u] |= 1<<v
        masks[v] |= 1<<u
    print(f"C4 count: {count_c4_bitmask(masks)}")
