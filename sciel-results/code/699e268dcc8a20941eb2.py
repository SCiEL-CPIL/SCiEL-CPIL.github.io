"""Improved program: construct() returns 91 edges on vertices 0..30 minimizing C4s.
Uses polarity graph cores with efficient delta-based simulated annealing and local search.
"""
import numpy as np
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s kill

    q = 5
    # Enumerate all points of PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Try multiple polarity forms (including non-diagonal symmetric forms)
    best_result = None
    best_total_c4 = float('inf')
    
    # Generate symmetric 3x3 matrices over GF(5) that are non-degenerate
    forms = []
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                forms.append((a, b, c, None))
    
    for a in [1, 2, 3, 4]:
        for b in [1, 2, 3, 4]:
            for c in [1, 2, 3, 4]:
                for d in [1, 2, 3, 4]:
                    forms.append((a, b, c, d))
    
    random.seed(42)
    random.shuffle(forms)
    
    cores_found = []
    for form in forms:
        if time.time() > deadline - 80:
            break
        
        a, b, c, d = form
        
        edges_core = []
        for i in range(31):
            for j in range(i+1, 31):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                if d is None:
                    val = (a*x1*x2 + b*y1*y2 + c*z1*z2) % 5
                else:
                    val = (a*x1*x2 + d*x1*y2 + d*y1*x2 + b*y1*y2 + c*z1*z2) % 5
                if val == 0:
                    edges_core.append([i, j])
        
        if len(edges_core) != 90:
            continue
        
        adj = [set() for _ in range(31)]
        for u, v in edges_core:
            adj[u].add(v)
            adj[v].add(u)
        
        best_edge = None
        best_add = float('inf')
        
        for u in range(31):
            for v in range(u+1, 31):
                if v in adj[u]:
                    continue
                add = 0
                for a2 in adj[u]:
                    for b2 in adj[v]:
                        if b2 in adj[a2]:
                            add += 1
                if add < best_add:
                    best_add = add
                    best_edge = [u, v]
        
        if best_edge is None:
            continue
        
        total_c4 = best_add
        
        if total_c4 < best_total_c4:
            best_total_c4 = total_c4
            best_result = edges_core + [best_edge]
        
        cores_found.append((edges_core, adj.copy()))
        
        if time.time() > deadline - 80:
            break
    
    if best_result is None:
        return _greedy_construct()
    
    # Use NumPy adjacency matrix for efficient operations
    # adj_mat[i, j] = 1 if edge exists
    adj_mat = np.zeros((31, 31), dtype=np.int8)
    for u, v in best_result:
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
    
    # Helper to count C4s using matrix multiplication
    # Number of C4s = 0.25 * sum_{i,j} (A^2)_{ij}^2 - ... 
    # Actually, standard formula: 
    # C4 count = 1/4 * ( sum_{i,j} (A^2)_{ij}^2 - sum_i (A^2)_{ii}^2 - 2*sum_i (A^2)_{ii} - 2*m )
    # Simpler: 
    # For each pair (i,j), common neighbors = (A^2)_{ij}.
    # C4s involving edge (i,j) is C(common, 2).
    # Total C4s = sum_{i<j} C((A^2)_{ij}, 2)
    
    def count_c4s(A):
        A2 = A @ A
        # A2[i,j] is number of common neighbors of i and j
        # Sum over all i<j of C(A2[i,j], 2)
        # C(k, 2) = k*(k-1)/2
        # We need to sum only for i<j
        # Since A2 is symmetric, we can take upper triangle
        upper = np.triu(A2, 1)
        return int(np.sum(upper * (upper - 1) // 2))
    
    current_c4 = count_c4s(adj_mat)
    best_c4 = current_c4
    best_adj = adj_mat.copy()
    
    # List of all possible edges (u, v) with u < v
    all_pairs = [(i, j) for i in range(31) for j in range(i+1, 31)]
    edge_indices = { (i, j): idx for idx, (i, j) in enumerate(all_pairs) }
    
    # Current edges set for quick lookup
    current_edges = set()
    for u, v in best_result:
        current_edges.add((min(u, v), max(u, v)))
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    no_improve = 0
    patience = 2000
    
    while time.time() < deadline - 20.0 and T > 0.1:
        if no_improve > patience:
            break
            
        # Pick a random edge to remove
        u, v = random.choice(list(current_edges))
        
        # Pick a random non-edge to add
        # To be efficient, we can just pick random pairs and check if they are non-edges
        # Or maintain a list of non-edges. Given 465 total pairs, 91 edges, ~374 non-edges.
        # Let's pick a random non-edge.
        added = False
        while not added:
            a, b = random.choice(all_pairs)
            if (a, b) not in current_edges:
                break
        
        # Calculate delta
        # Delta = C4s after - C4s before
        # C4s after = C4s before - C4s_using(u,v) + C4s_using(a,b)_in_new_graph
        
        # Helper: count C4s using edge (x,y) in graph A
        def c4s_using_edge(A, x, y):
            # Common neighbors of x and y
            # Neighbors of x: A[x]
            # Neighbors of y: A[y]
            # Common: A[x] & A[y]
            # But we need to count C4s that USE the edge (x,y).
            # A C4 is x-y-z-w-x.
            # z is a neighbor of y and w is a neighbor of x?
            # Actually, if x-y is an edge, a C4 containing it is x-y-z-w-x.
            # z must be adjacent to y, w must be adjacent to x.
            # And z-w must be an edge? No, z and w are connected via the rest of the cycle.
            # Wait, the standard definition of C4 in simple graph:
            # A cycle of length 4.
            # If x-y is an edge, the other two edges must connect x to some w and y to some z, and w-z must be an edge?
            # No, the cycle is x-y-z-w-x.
            # Edges: (x,y), (y,z), (z,w), (w,x).
            # So z is a neighbor of y (z != x), w is a neighbor of x (w != y).
            # And (z,w) must be an edge.
            # So we need to count pairs (z, w) such that:
            # 1. z in N(y), z != x
            # 2. w in N(x), w != y
            # 3. (z, w) is an edge in the graph.
            
            # Note: If z == w, then x-y-z-x is a triangle, not a C4.
            # So z != w.
            
            cnt = 0
            ny = np.where(A[y] == 1)[0]
            nx = np.where(A[x] == 1)[0]
            
            # Filter out x from ny and y from nx
            ny = [n for n in ny if n != x]
            nx = [n for n in nx if n != y]
            
            # For each z in ny, count w in nx such that A[z, w] == 1 and z != w
            # We can do this with matrix multiplication or loops.
            # Since degrees are small (~5-10), loops are fine.
            for z in ny:
                for w in nx:
                    if z != w and A[z, w] == 1:
                        cnt += 1
            return cnt
        
        c4s_before = current_c4
        c4s_uv = c4s_using_edge(adj_mat, u, v)
        
        # Temporarily remove (u, v) to calculate c4s_ab in the modified graph?
        # No, c4s_using_edge(a, b) depends on the graph structure.
        # If we add (a, b), the new C4s are those that use (a, b) in the graph WITHOUT (a, b) but WITH (u, v) removed?
        # Actually, the change is:
        # New Graph = Old Graph - {(u,v)} + {(a,b)}
        # C4s(New) = C4s(Old) - C4s_using(u,v in Old) + C4s_using(a,b in New)
        
        # To calculate C4s_using(a,b in New):
        # We need the adjacency of the New graph.
        # Let's create a temporary view or just calculate it carefully.
        
        # Let's just compute the delta directly by checking all potential C4s affected.
        # Affected C4s must contain (u,v) or (a,b).
        
        # This is getting complicated. Let's use a simpler heuristic for delta estimation or just accept the cost of recounting if we do it smartly.
        # Given 31 vertices, recounting all C4s takes ~31^2 * deg^2 which is small.
        # 31*31*10*10 = 96k ops. 100k steps * 96k = 10^10 ops. Too slow.
        
        # We need efficient delta.
        # Let's stick to the previous logic but fix the data structure issues.
        # The previous logic was:
        # d_rem = -c4_contribution(u, v, adj)
        # d_add = c4_contribution(a3, b3, adj)  <-- This is WRONG if adj still has (u,v) and doesn't have (a,b).
        # The C4s added by (a,b) might depend on (u,v) being present or not?
        # If a C4 uses both (u,v) and (a,b), it's complicated.
        # However, if we assume local moves, the overlap is rare.
        
        # Let's try a robust but slightly slower delta:
        # 1. Compute C4s using (u,v) in current graph.
        # 2. Remove (u,v) from a copy of adj_mat? No, too slow.
        # 3. Compute C4s using (a,b) in graph where (u,v) is removed.
        
        # Let's implement a function that calculates C4s using edge (x,y) given the current adj_mat,
        # but allowing us to override the state of edge (u,v).
        
        def c4s_using_edge_with_override(A, x, y, rem_u, rem_v, add_u, add_v):
            # This is getting messy.
            # Let's just do a full recount every K steps? No.
            
            # Alternative: 
            # C4s containing (u,v) are destroyed.
            # C4s containing (a,b) are created.
            # C4s containing BOTH (u,v) and (a,b) are:
            # - Destroyed by removing (u,v)
            # - Created by adding (a,b)
            # Net change for such C4s is 0.
            # So we can just calculate:
            # Delta = - C4s_using(u,v) + C4s_using(a,b)
            # Where C4s_using(a,b) is calculated in the graph where (u,v) is REMOVED?
            # If a C4 uses (a,b) and (u,v), it is counted in C4s_using(u,v) and C4s_using(a,b).
            # If we remove (u,v), that C4 is gone.
            # If we add (a,b), that C4 is back.
            # So yes, if we calculate C4s_using(a,b) in the graph WITHOUT (u,v), we are correct.
            
            # So:
            # 1. Calculate C4s_using(u, v) in current graph.
            # 2. Calculate C4s_using(a, b) in graph where (u, v) is removed.
            
            pass
        
        # Let's implement this.
        
        # 1. C4s using (u,v)
        c4s_uv = c4s_using_edge(adj_mat, u, v)
        
        # 2. C4s using (a,b) in graph without (u,v)
        # We can simulate this by checking if a candidate C4 for (a,b) uses (u,v).
        # A C4 using (a,b) is a-b-c-d-a.
        # It uses (u,v) if {c,d} contains u and v? Or if c==u and d==v?
        # Since (u,v) is an edge, for the C4 to use it, the edge (u,v) must be one of the other edges.
        # The edges are (a,b), (b,c), (c,d), (d,a).
        # So (u,v) must be (b,c), (c,d), or (d,a).
        
        c4s_ab = 0
        nb = np.where(adj_mat[b] == 1)[0]
        na = np.where(adj_mat[a] == 1)[0]
        
        # Filter out a from nb and b from na
        nb = [n for n in nb if n != a]
        na = [n for n in na if n != b]
        
        for c in nb:
            for d in na:
                if c == d:
                    continue
                # Check if (c, d) is an edge
                if adj_mat[c, d] == 0:
                    continue
                
                # Check if this C4 (a-b-c-d-a) uses the edge (u, v)
                # The edges are (a,b), (b,c), (c,d), (d,a).
                # (u,v) is used if:
                # (u,v) == (b,c) or (v,u) == (b,c)
                # or (u,v) == (c,d) or (v,u) == (c,d)
                # or (u,v) == (d,a) or (v,u) == (d,a)
                # Note: (a,b) is the edge we are adding, so it's not (u,v) because (u,v) is being removed and (a,b) is not currently an edge.
                
                uses_uv = False
                if (b == u and c == v) or (b == v and c == u):
                    uses_uv = True
                if (c == u and d == v) or (c == v and d == u):
                    uses_uv = True
                if (d == u and a == v) or (d == v and a == u):
                    uses_uv = True
                
                if not uses_uv:
                    c4s_ab += 1
        
        delta = c4s_ab - c4s_uv
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Apply swap
            adj_mat[u, v] = 0
            adj_mat[v, u] = 0
            current_edges.discard((u, v))
            
            adj_mat[a, b] = 1
            adj_mat[b, a] = 1
            current_edges.add((a, b))
            
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj_mat.copy()
                no_improve = 0
            else:
                no_improve += 1
        
        T *= cooling
    
    # Reconstruct best edges from best_adj
    final_edges = []
    for i in range(31):
        for j in range(i+1, 31):
            if best_adj[i, j] == 1:
                final_edges.append([i, j])
    
    # Validate
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    
    return final_edges

def _greedy_construct():
    """Fallback greedy construction."""
    random.seed(42)
    
    edges = []
    adj = np.zeros((31, 31), dtype=np.int8)
    
    remaining = []
    for i in range(31):
        for j in range(i+1, 31):
            remaining.append([i, j])
    
    random.shuffle(remaining)
    
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if adj[u, v] == 1:
                continue
            add = 0
            # Count common neighbors of u and v that are connected?
            # No, add edge (u,v). C4s added = pairs (a,b) such that a in N(u), b in N(v), a!=b, and (a,b) is edge.
            # Wait, C4 is u-v-b-a-u.
            # Edges: (u,v), (v,b), (b,a), (a,u).
            # So b in N(v), a in N(u), and (a,b) is edge.
            # And a != b.
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            for a in nu:
                for b in nv:
                    if a != b and adj[a, b] == 1:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u, v] = 1
        adj[v, u] = 1
        remaining.remove(best_e)
    
    while len(edges) < 91 and remaining:
        for e in remaining:
            u, v = e
            if adj[u, v] == 0:
                edges.append(e)
                adj[u, v] = 1
                adj[v, u] = 1
                remaining.remove(e)
                break
    
    return edges[:91]
