import numpy as np
import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    # 1. Generate PG(2,5) points
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == y == z == 0:
                    continue
                # Normalize: first non-zero coord to 1
                if x != 0:
                    inv = pow(x, 3, 5)  # x^(p-2) = x^3 mod 5
                    key = (1, (y * inv) % 5, (z * inv) % 5)
                elif y != 0:
                    inv = pow(y, 3, 5)
                    key = (0, 1, (z * inv) % 5)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n = 31
    assert len(points) == n

    # 2. Define Polarity Graph
    # Quadratic form Q(x,y,z) = x^2 + y^2 + z^2 - x*y - y*z - z*x ? 
    # Or standard: x^2 + y^2 + z^2 = 0? No, that's not a polarity.
    # A polarity graph is defined by a bilinear form B(u,v) = 0.
    # For PG(2,q) with q odd, a standard polarity is given by the matrix:
    # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]? No, that's symmetric.
    # B(u,v) = u1v1 + u2v2 + u3v3. If u=v, Q(u) = sum ui^2.
    # In PG(2,5), the number of absolute points (Q(u)=0) determines the self-loops.
    # We want a graph without self-loops, so we ignore self-loops.
    # The number of edges in the polarity graph is (q(q-1)/2) + (number of absolute points).
    # For q=5, q(q-1)/2 = 10. Total edges = 10 + abs_points.
    # We need 90 edges. So abs_points must be 80? No, max vertices is 31.
    # Wait, the formula for edges in a polarity graph of PG(2,q) is:
    # E = q(q-1)/2 + (q+1) = (q^2-q+2q+2)/2 = (q^2+q+2)/2?
    # Let's check: q=5. (25+5+2)/2 = 32? No.
    # Standard result: ex(n, C4) for n=q^2+q+1 is q(q+1)/2 + (q+1) = (q^2+3q+2)/2?
    # For q=5: (25+15+2)/2 = 21? No.
    # Let's just calculate it.
    
    # Bilinear form B(u,v) = u1*v1 + u2*v2 + u3*v3
    # Edge if B(u,v) == 0 and u != v.
    
    edges_core = []
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i + 1, n):
            u = np.array(points[i])
            v = np.array(points[j])
            # B(u,v) = dot(u,v) mod 5
            if np.dot(u, v) % 5 == 0:
                edges_core.append((i, j))
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Check edge count
    num_edges_core = len(edges_core)
    # For q=5, expected edges is 90.
    # Let's verify.
    # If num_edges_core != 90, we might have a different polarity or need to adjust.
    # Actually, the standard polarity on PG(2,5) yields 90 edges.
    
    if num_edges_core != 90:
        # Fallback: try another form or just use greedy
        # But usually this specific form works.
        pass

    def count_c4(adj_matrix):
        # adj_matrix is 31x31 int8
        # M = A @ A
        # M[i,j] is number of common neighbors
        # C4s = 1/4 * sum_{i<j} M[i,j]*(M[i,j]-1)/2
        # Note: M is symmetric.
        M = adj_matrix @ adj_matrix
        total = 0
        # Vectorized sum over upper triangle
        # M[i,j] for i<j
        iu, ju = np.triu_indices(n, k=1)
        vals = M[iu, ju]
        # Only count if vals >= 2
        mask = vals >= 2
        if np.any(mask):
            v_masked = vals[mask]
            total = np.sum(v_masked * (v_masked - 1) // 2)
        return total // 4

    # 3. Find best edge to add to core
    # We have 90 edges. We need 1 more.
    # Total possible edges: 31*30/2 = 465.
    # Non-edges: 465 - 90 = 375.
    
    current_c4 = 0 # Core is C4-free
    best_edge = None
    best_add_c4 = float('inf')
    
    # Precompute common neighbors for speed?
    # M_core = adj @ adj
    M_core = adj @ adj
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # Adding edge (i,j) creates C4s for each pair of common neighbors
                # Number of new C4s = C(common_neighbors, 2)
                # common_neighbors = M_core[i, j]
                cn = M_core[i, j]
                if cn >= 2:
                    new_c4s = cn * (cn - 1) // 2
                else:
                    new_c4s = 0
                
                if new_c4s < best_add_c4:
                    best_add_c4 = new_c4s
                    best_edge = (i, j)
    
    if best_edge is None:
        # Should not happen
        best_edge = (0, 1)
        # Ensure it's a non-edge
        while adj[best_edge[0], best_edge[1]] == 1:
             best_edge = (best_edge[0], best_edge[1]+1)
             if best_edge[1] >= n:
                 best_edge = (best_edge[0]+1, 0)
                 if best_edge[0] >= n: break
                 while best_edge[0] == best_edge[1]:
                     best_edge = (best_edge[0], best_edge[1]+1)

    # Add the best edge
    u, v = best_edge
    adj[u, v] = 1
    adj[v, u] = 1
    edges = edges_core + [(u, v)]
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edges = [list(e) for e in edges]
    
    # 4. Local Search
    # We will use a combination of 1-swaps and 2-swaps.
    # 1-swap: Remove edge e1, add non-edge e2.
    # 2-swap: Remove e1, e2, add f1, f2.
    
    # To speed up, we can precompute the effect of removing an edge.
    # But 31 is small.
    
    random.seed(42)
    
    # List of all edges and non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(edges)
    non_edge_set = set(all_pairs) - edge_set
    
    # Helper to get delta C4 for removing an edge (u,v)
    # C4s destroyed = number of paths of length 3 from u to v in G - (u,v)
    # which is equal to M[u,v] in current graph?
    # No, M[u,v] is common neighbors. Each pair of common neighbors forms a C4 containing (u,v).
    # So C4s destroyed by removing (u,v) = C(M[u,v], 2).
    
    # Helper to get delta C4 for adding an edge (u,v) to G
    # C4s created = C(M[u,v], 2).
    
    # So for a 1-swap (remove e_rm, add e_add):
    # Delta = C(M[e_add], 2) - C(M[e_rm], 2)
    # Note: M changes after removal/addition, but for small changes, this is a good approximation?
    # Actually, M[e_rm] decreases by 1 for neighbors of u and v? No.
    # M[i,j] is number of common neighbors.
    # If we remove (u,v), M[u,v] becomes 0? No, M[u,v] was the count of common neighbors.
    # The C4s using edge (u,v) are exactly the pairs of common neighbors.
    # So yes, Delta = C(M_new_add, 2) - C(M_old_rm, 2) is exact if the added edge doesn't affect the common neighbor count of the removed edge (which it doesn't, as they are distinct edges in the graph structure, though they share vertices).
    # Wait, if we remove (u,v) and add (x,y), and (x,y) is adjacent to u or v, the M values change.
    # So the simple formula is an approximation.
    # However, for hill climbing, we can just recompute C4 count if the approximation suggests improvement, or just recompute for all candidates if fast enough.
    
    # Recomputing C4 count is O(n^3) ~ 31^3 = 30k ops.
    # Number of 1-swaps: 91 edges * 374 non-edges ~ 34k swaps.
    # 34k * 30k = 1e9 ops. Too slow in Python.
    # But we can use the approximation to filter, then verify.
    
    # Let's use the approximation to find candidates, then verify top K.
    
    def get_c4_delta_1swap(adj, M, e_rm, e_add):
        # e_rm = (u,v), e_add = (x,y)
        u, v = e_rm
        x, y = e_add
        # Approximation
        c4_rm = M[u, v] * (M[u, v] - 1) // 2
        c4_add = M[x, y] * (M[x, y] - 1) // 2
        return c4_add - c4_rm

    # We will iterate until time runs out.
    # Use Simulated Annealing to escape local minima.
    
    T = 5.0
    cooling = 0.995
    
    # Precompute M
    M = adj @ adj
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # 1. Try 1-swaps
        # Shuffle edges and non-edges to avoid bias
        edges_list = list(edge_set)
        non_edges_list = list(non_edge_set)
        random.shuffle(edges_list)
        random.shuffle(non_edges_list)
        
        # Limit checks per iteration to allow time for 2-swaps later?
        # Just do all 1-swaps if fast enough with approximation.
        
        best_delta = 0
        best_move = None
        
        for e_rm in edges_list:
            # Check time
            if time.time() > deadline - 5.0:
                break
            for e_add in non_edges_list:
                if e_add == e_rm: continue
                delta = get_c4_delta_1swap(adj, M, e_rm, e_add)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (e_rm, e_add)
                    if delta < -10: # Good improvement, take it? Or keep looking?
                         # Keep looking for global best in this pass
                         pass
        
        if best_move and best_delta < 0:
            e_rm, e_add = best_move
            # Apply move
            u, v = e_rm
            x, y = e_add
            adj[u, v] = 0
            adj[v, u] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            non_edge_set.remove(e_add)
            non_edge_set.add(e_rm)
            
            # Update M
            M = adj @ adj
            current_c4 = count_c4(adj)
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = [list(e) for e in edge_set]
                improved = True
        else:
            # No improving 1-swap found. Try 2-swaps or cool down.
            # 2-swaps are more expensive.
            # Try a random 2-swap with SA acceptance.
            if T > 0.1:
                # Pick 2 edges to remove
                if len(edge_set) >= 2:
                    e_rm1, e_rm2 = random.sample(list(edge_set), 2)
                    # Pick 2 non-edges to add
                    if len(non_edge_set) >= 2:
                        e_add1, e_add2 = random.sample(list(non_edge_set), 2)
                        
                        # Approximate delta
                        d1 = get_c4_delta_1swap(adj, M, e_rm1, e_add1)
                        d2 = get_c4_delta_1swap(adj, M, e_rm2, e_add2)
                        # Interaction is ignored in approx
                        delta_approx = d1 + d2
                        
                        if delta_approx < 0 or random.random() < np.exp(-max(0, delta_approx) / T):
                            # Apply
                            for u, v in [e_rm1, e_rm2]:
                                adj[u, v] = 0
                                adj[v, u] = 0
                            for u, v in [e_add1, e_add2]:
                                adj[u, v] = 1
                                adj[v, u] = 1
                            
                            edge_set.remove(e_rm1)
                            edge_set.remove(e_rm2)
                            edge_set.add(e_add1)
                            edge_set.add(e_add2)
                            
                            non_edge_set.remove(e_add1)
                            non_edge_set.remove(e_add2)
                            non_edge_set.add(e_rm1)
                            non_edge_set.add(e_rm2)
                            
                            M = adj @ adj
                            current_c4 = count_c4(adj)
                            
                            if current_c4 < best_c4:
                                best_c4 = current_c4
                                best_edges = [list(e) for e in edge_set]
                                improved = True
            T *= cooling
        
        if not improved and T < 0.1:
             # Stuck and cold. Try random restart? Or just break.
             # Given time constraints, break.
             pass

    # Convert back to list of lists
    return [list(e) for e in best_edges]
