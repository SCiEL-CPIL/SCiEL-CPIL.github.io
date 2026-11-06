import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'n': 39, 'edges': []}
    best_n = 39
    
    # Helper to check if adding edge (u, v) maintains C4-free property
    # A C4 exists if any pair has > 1 common neighbor.
    # Adding (u, v) increases common neighbor count for:
    # 1. Pairs (u, w) where w is neighbor of v: c(u, w) increases by 1 (due to v)
    # 2. Pairs (v, w) where w is neighbor of u: c(v, w) increases by 1 (due to u)
    # 3. Pair (u, v): c(u, v) is sum of adj[u][x]*adj[v][x]. Adding edge (u,v) does not change this sum 
    #    because x cannot be u or v in the sum for common neighbors of u and v? 
    #    Wait, common neighbors of u and v are w != u, v such that (u,w) and (v,w) are edges.
    #    So c(u,v) is unchanged by adding (u,v).
    #    
    #    So we just need to ensure that for all w in N(v) (excluding u), c(u, w) < 1 (i.e., 0) before adding.
    #    And for all w in N(u) (excluding v), c(v, w) < 1 (i.e., 0) before adding.
    #    If c(u, w) >= 1, adding (u, v) makes c(u, w) >= 2, creating a C4.
    
    def can_add_edge(adj, n, u, v):
        # Check neighbors of v
        # We need c(u, w) == 0 for all w in N(v) \ {u}
        # c(u, w) = dot(adj[u], adj[w])
        # Since n is small, we can just iterate.
        for w in range(n):
            if w == u or w == v: continue
            if adj[v][w] == 1:
                # Check if u and w have any common neighbor
                # If they do, adding (u,v) creates a C4 u-v-w-x-u
                # where x is the common neighbor.
                if np.dot(adj[u], adj[w]) > 0:
                    return False
            if adj[u][w] == 1:
                # Check if v and w have any common neighbor
                # If they do, adding (u,v) creates a C4 u-v-w-x-u
                if np.dot(adj[v], adj[w]) > 0:
                    return False
        return True

    def try_construct(n, seed):
        random.seed(seed)
        target_deg = n - 39
        
        # Strategy: Start with a random graph that is likely C4-free (low density) 
        # or use a structured approach? 
        # The prompt mentions Polarity Graph of PG(2,7) as a strong idea.
        # PG(2,7) has 57 points. It is a C4-free graph with degree 8.
        # We need n <= 47. 
        # If we take a subgraph of the Polarity Graph, it remains C4-free.
        # The Polarity Graph of PG(2,7) is a 57-vertex graph, 8-regular.
        # If we take any 47 vertices, the induced subgraph is C4-free.
        # The degrees in the induced subgraph will be <= 8.
        # For n=47, target_deg = 8. 
        # For n=46, target_deg = 7.
        # If we can find a subset of vertices in PG(2,7) such that the induced subgraph has min degree >= target, we win.
        
        # Let's try to construct the Polarity Graph of PG(2,7) first.
        # PG(2, q) points are 1D subspaces of F_q^3.
        # For q=7, there are (7^3 - 1)/(7-1) = 342/6 = 57 points.
        # Incidence: Two points are adjacent if they are orthogonal w.r.t a polarity.
        # A standard polarity in PG(2,7) can be defined by a symmetric matrix.
        # Let's use the standard unit polarity: x_0^2 + x_1^2 + x_2^2 = 0?
        # In F_7, squares are 0, 1, 2, 4.
        # Let's generate the points.
        
        def generate_polarity_graph(q):
            # Generate all points in PG(2, q)
            points = []
            # Normalize points so first non-zero coordinate is 1
            for x in range(q):
                for y in range(q):
                    for z in range(q):
                        if x == 0 and y == 0 and z == 0: continue
                        # Normalize
                        if x != 0:
                            # x is already 1? No, x can be 1..q-1. 
                            # We want unique reps.
                            # If x != 0, multiply by x^{-1} to make it 1.
                            inv_x = pow(x, -1, q)
                            px = 1
                            py = (y * inv_x) % q
                            pz = (z * inv_x) % q
                            points.append((px, py, pz))
                        elif y != 0:
                            inv_y = pow(y, -1, q)
                            px = 0
                            py = 1
                            pz = (z * inv_y) % q
                            points.append((px, py, pz))
                        else:
                            # z != 0
                            inv_z = pow(z, -1, q)
                            px = 0
                            py = 0
                            pz = 1
                            points.append((px, py, pz))
            
            # Remove duplicates (should be unique by construction)
            unique_points = list(set(points))
            # Sort for determinism
            unique_points.sort()
            
            # Define polarity. 
            # A self-adjoint bilinear form.
            # Let's use the diagonal form: x_0^2 + x_1^2 + x_2^2 = 0?
            # In F_7, 1^2+1^2+1^2 = 3 != 0.
            # We need a non-degenerate symmetric matrix A.
            # Let A = I. Then u . v = u0v0 + u1v1 + u2v2.
            # Two points are adjacent if u . v = 0.
            
            n_pts = len(unique_points)
            adj = np.zeros((n_pts, n_pts), dtype=np.int8)
            
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    u = unique_points[i]
                    v = unique_points[j]
                    dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
                    if dot == 0:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        
            return unique_points, adj

        try:
            pts, full_adj = generate_polarity_graph(7)
            n_full = len(pts)
            
            # We want a subset of size n such that min degree in induced subgraph >= target_deg.
            # Greedy selection:
            # Start with all vertices.
            # If n_full > n, we need to remove vertices.
            # Removing a vertex reduces degrees of its neighbors.
            # We want to remove vertices that are "least useful" or have low degree?
            # Actually, we want to KEEP vertices that have high degree in the final subgraph.
            # Or, we can try to find a subgraph of size n with min degree >= target.
            
            if n_full >= n:
                # Try to find a subset of size n
                # Heuristic: Start with all vertices, iteratively remove vertex with lowest degree in current subgraph?
                # No, that might drop min degree.
                # Better: Start with a random subset of size n?
                # Or: Start with all, and if min degree is too high (it's 8), we can just pick any n vertices?
                # If we pick any n vertices from an 8-regular graph, the degrees in the induced subgraph will be <= 8.
                # They could be as low as 0.
                # We need min degree >= target.
                
                # Let's try random subsets of size n.
                # Number of vertices to remove: n_full - n.
                # For n=47, remove 10.
                # For n=46, remove 11.
                
                # Strategy: Randomly select n vertices. Check min degree.
                # Repeat with different seeds?
                # We have `seed` available.
                
                for attempt in range(50): # Try 50 random subsets
                    if time.time() > deadline:
                        return None
                    
                    # Select n random vertices
                    if n_full == n:
                        subset_indices = list(range(n_full))
                    else:
                        subset_indices = random.sample(range(n_full), n)
                    
                    # Map old indices to new indices
                    old_to_new = {old: new for new, old in enumerate(subset_indices)}
                    
                    # Build induced subgraph
                    adj_sub = np.zeros((n, n), dtype=np.int8)
                    for new_i, old_i in enumerate(subset_indices):
                        for new_j, old_j in enumerate(subset_indices):
                            if new_i < new_j:
                                if full_adj[old_i][old_j]:
                                    adj_sub[new_i][new_j] = 1
                                    adj_sub[new_j][new_i] = 1
                    
                    # Check min degree
                    degs = adj_sub.sum(axis=1)
                    if np.min(degs) >= target_deg:
                        # Valid!
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj_sub[i][j]:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
            
            # If polarity graph subset didn't work, fall back to random + repair
            # But polarity graph is 8-regular. 
            # For n=47, target=8. We need an induced subgraph that is 8-regular? 
            # That means we need to keep ALL neighbors of every vertex? 
            # If we remove any vertex, its neighbors lose 1 degree.
            # So for n=47 (target 8), we must keep a subset where no vertex lost a neighbor.
            # This implies the subset is a union of connected components? No.
            # It implies that for every vertex in the subset, all its 8 neighbors in the full graph are ALSO in the subset.
            # This is a "closed" set under neighborhood.
            # In a connected 8-regular graph, the only such sets are the whole graph or empty?
            # If the graph is connected, and S is a subset such that N(v) subset S for all v in S.
            # Then S must be the whole graph if the graph is connected.
            # PG(2,7) polarity graph is connected.
            # So for n=47, target=8, we CANNOT find a proper induced subgraph with min degree 8.
            # We MUST use n=57? No, n <= 47.
            # So for n=47, we need min degree >= 8.
            # If we take a subset of size 47 from the 57-vertex 8-regular graph, 
            # the max possible min degree is 8, but only if the subset is the whole graph (impossible) 
            # or if the graph has a component of size 47? Unlikely.
            # So for n=47, we likely need a different graph or a repair step.
            
            # However, for n=46, target=7.
            # We can remove 11 vertices.
            # If we remove vertices such that each remaining vertex loses at most 1 neighbor, min degree >= 7.
            # This is a vertex deletion problem.
            
        except Exception:
            pass

        # Fallback: Random graph + Repair
        # Use the logic from the current program but maybe optimized.
        
        densities = [0.25, 0.22, 0.20, 0.18, 0.16, 0.14]
        
        for p in densities:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Initial check
            degs = adj.sum(axis=1)
            if np.min(degs) < target_deg:
                continue
            
            # Greedy removal
            for step in range(1000):
                if time.time() > deadline:
                    return None
                
                common = adj @ adj
                tmp = common.copy()
                np.fill_diagonal(tmp, 0)
                
                if tmp.max() < 2:
                    if np.min(adj.sum(axis=1)) >= target_deg:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        break
                
                indices = np.triu_indices(n, k=1)
                vals = tmp[indices]
                viol_mask = vals >= 2
                if not np.any(viol_mask):
                    break
                
                viol_positions = np.where(viol_mask)[0]
                pos = random.choice(viol_positions)
                u, v = indices[0][pos], indices[1][pos]
                
                commons = np.where((adj[u] & adj[v]) == 1)[0]
                if len(commons) == 0:
                    continue
                
                w = random.choice(commons.tolist())
                
                deg_u = adj[u].sum()
                deg_v = adj[v].sum()
                
                if deg_u > deg_v:
                    x = u
                elif deg_v > deg_u:
                    x = v
                else:
                    x = u if random.random() < 0.5 else v
                
                adj[x][w] = 0
                adj[w][x] = 0
                
                if np.min(adj.sum(axis=1)) < target_deg:
                    break
        
        return None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        found = False
        # Try polarity graph strategy first for specific n if applicable
        # The polarity graph loop is inside try_construct, so it will be tried for each seed.
        # We can increase the number of seeds to improve the chance of finding a good subset for n=46.
        
        for seed in range(500):
            if time.time() > deadline:
                break
            res = try_construct(n, seed)
            if res is not None:
                if n > best_n:
                    best_n = n
                    best_result = res
                found = True
                if n == 47:
                    return best_result
                break
        
        if found:
            continue
        else:
            pass
    
    return best_result
