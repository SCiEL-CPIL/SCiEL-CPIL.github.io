"""Program for fewest four-cycles in graphs with 31 vertices and 91 edges."""
import numpy as np
from scipy.sparse import coo_matrix
import random
import time

def construct():
    """
    Construct a graph with 31 vertices and exactly 91 edges minimizing the number of C4 subgraphs.
    
    Approach:
    1. Build a C4-free graph with 90 edges using a polarity graph construction on PG(2,5).
    2. Add one edge carefully to minimize the number of new C4s created.
    3. Perform local search (edge swaps) to further minimize C4 count.
    """
    n = 31
    target_edges = 91
    
    # Step 1: Build a C4-free base graph with 90 edges
    # Use a polarity graph construction for q=5
    # The vertices are the 31 points of PG(2,5)
    # We represent points as homogeneous coordinates [x,y,z] over GF(5)
    
    # Generate all points in PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                pt = [x, y, z]
                for i in range(3):
                    if pt[i] != 0:
                        inv = pow(pt[i], 3, 5)  # inverse in GF(5)
                        pt = [(pt[j] * inv) % 5 for j in range(3)]
                        break
                if tuple(pt) not in seen:
                    seen.add(tuple(pt))
                    points.append(pt)
    
    # Should have 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define a polarity: A point p = (x,y,z) is incident to point q = (u,v,w) 
    # if x*w + y*u + z*v = 0 in GF(5) (symmetric bilinear form)
    # This defines a symmetric relation, so we need to handle self-incidence carefully
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            # Check polarity condition: p[0]*q[2] + p[1]*q[0] + p[2]*q[1] == 0 mod 5
            val = (p[0] * q[2] + p[1] * q[0] + p[2] * q[1]) % 5
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Count edges in base graph
    edge_count = sum(adj[i, j] for i in range(n) for j in range(i + 1, n))
    
    # If we don't have exactly 90 edges, adjust
    # For q=5, the polarity graph should have (q^3 + q^2 + q)/2 = (125+25+5)/2 = 77.5? 
    # Let me recalculate. Actually for a polarity graph on PG(2,q) with q odd,
    # the number of edges depends on the specific polarity.
    
    # Let's just use whatever we got and add/remove edges to reach 90
    if edge_count < 90:
        # Add random non-edges
        random.seed(42)
        added = 0
        attempts = 0
        while added < 90 - edge_count and attempts < 10000:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                added += 1
            attempts += 1
    elif edge_count > 90:
        # Remove random edges
        random.seed(42)
        removed = 0
        attempts = 0
        while removed < edge_count - 90 and attempts < 10000:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and adj[i, j] == 1:
                adj[i, j] = 0
                adj[j, i] = 0
                removed += 1
            attempts += 1
    
    # Now we have 90 edges. Add one more edge to get 91.
    # To minimize C4s, add an edge between two vertices with minimum common neighbors
    min_common = float('inf')
    best_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                common = np.sum(adj[i] & adj[j])
                if common < min_common:
                    min_common = common
                    best_pair = (i, j)
    
    if best_pair is not None:
        i, j = best_pair
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Now we have 91 edges. Perform local search to minimize C4 count.
    # Efficient local search using delta calculations
    
    def get_common_neighbors_count(u, v):
        return np.sum(adj[u] & adj[v])
    
    def delta_add_edge(u, v):
        # Calculate change in C4 count if we add edge (u, v)
        # Assumes (u, v) is currently NOT an edge
        c_uv = get_common_neighbors_count(u, v)
        
        # Term 1: New C4s containing (u, v)
        delta = c_uv * (c_uv - 1) // 2
        
        # Term 2: Increase in C4s for pairs involving u or v
        # For each neighbor y of v, the pair (u, y) gains v as a common neighbor.
        # The count for (u, y) increases by 1.
        # The change in C4 count for pair (u, y) is (c_uy + 1 choose 2) - (c_uy choose 2) = c_uy
        # So we sum c_uy for all y in N(v)
        
        # Note: u is not in N(v) since (u,v) is not an edge.
        # v is not in N(u) since (u,v) is not an edge.
        
        # We need sum of common neighbors of u and y for all y in N(v)
        # This is sum_{y in N(v)} |N(u) & N(y)|
        
        # Efficient calculation:
        # N(u) & N(y) is the set of common neighbors.
        # Sum of sizes of these sets.
        
        # We can do this by iterating neighbors of v
        sum_uy = 0
        for y in np.nonzero(adj[v])[0]:
            if y == u: continue # should not happen
            sum_uy += np.sum(adj[u] & adj[y])
            
        # Similarly for x in N(u), sum c_vx
        sum_vx = 0
        for x in np.nonzero(adj[u])[0]:
            if x == v: continue # should not happen
            sum_vx += np.sum(adj[x] & adj[v])
            
        delta += sum_uy + sum_vx
        return delta
    
    def delta_remove_edge(u, v):
        # Calculate change in C4 count if we remove edge (u, v)
        # Assumes (u, v) is currently AN edge
        # This is the negative of delta_add_edge on the graph where (u,v) is removed?
        # No, it's symmetric. Removing (u,v) decreases C4 count by the same amount that adding it would increase it,
        # calculated on the graph *without* the edge?
        # Actually, let's just derive it.
        # Removing (u,v) removes C4s containing (u,v). Count: C(c_uv, 2) where c_uv is common neighbors in current graph.
        # But wait, if (u,v) is an edge, u is in N(v) and v is in N(u).
        # Common neighbors of u and v in current graph: N(u) & N(v).
        # Let c = |N(u) & N(v)|.
        # C4s containing (u,v) are u-w-v-x-u. w, x in N(u) & N(v).
        # So we lose C(c, 2) C4s.
        
        # Also, for pairs (u, y) where y in N(v) \ {u}, u and y lose v as a common neighbor.
        # c_uy decreases by 1. Change in C4 count for pair (u,y) is C(c_uy-1, 2) - C(c_uy, 2) = -c_uy + 1.
        # So we sum (-c_uy + 1) for y in N(v) \ {u}.
        
        c_uv = get_common_neighbors_count(u, v)
        delta = - (c_uv * (c_uv - 1) // 2)
        
        # Sum for y in N(v) excluding u
        for y in np.nonzero(adj[v])[0]:
            if y == u: continue
            c_uy = get_common_neighbors_count(u, y)
            delta += -c_uy + 1
            
        # Sum for x in N(u) excluding v
        for x in np.nonzero(adj[u])[0]:
            if x == v: continue
            c_vx = get_common_neighbors_count(x, v)
            delta += -c_vx + 1
            
        return delta

    # Initial C4 count
    current_c4 = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = np.sum(adj[i] & adj[j])
            current_c4 += common * (common - 1) // 2

    random.seed(42)
    no_improvement = 0
    max_iterations = 100000
    iteration = 0
    
    # Precompute list of edges and non-edges for faster random selection?
    # Or just generate on the fly.
    
    while no_improvement < 20000 and iteration < max_iterations:
        iteration += 1
        
        # Pick a random edge to remove
        # To avoid O(n^2) scan, we can keep a list of edges, but it changes.
        # With n=31, O(n^2) is ~900, which is fast.
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1]
        if len(edges) != 91:
            break
        a, b = random.choice(edges)
        
        # Pick a random non-edge to add
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
        c, d = random.choice(non_edges)
        
        if (a == c and b == d) or (a == d and b == c):
            continue
        
        # Calculate delta for removing (a,b) and adding (c,d)
        # Note: The deltas are not independent if the edges interact (share vertices).
        # If {a,b} and {c,d} are disjoint, delta_total = delta_remove(a,b) + delta_add(c,d).
        # If they share a vertex, it's more complex.
        # Given the small size, let's just do the swap and recalculate if they share vertices?
        # Or just always recalculate C4 count? No, that's slow.
        
        # Let's check if they share vertices
        if a in (c, d) or b in (c, d):
            # Complex case: just do the swap and recount?
            # Recount is O(n^2) = 900 ops.
            # Delta calc is O(n * deg) ~ 30 * 10 = 300 ops.
            # So even recounting is comparable to delta calc for complex cases.
            # But simple cases are much faster with delta.
            
            # Let's just do the swap and recount for simplicity in complex cases?
            # Or better: implement a general delta function?
            # For now, if they share vertices, we'll just do the swap and full recount.
            # This happens with probability ~ (2*30/465) ~ 10% of the time?
            # So 90% of the time we use fast delta.
            
            adj[a, b] = 0
            adj[b, a] = 0
            adj[c, d] = 1
            adj[d, c] = 1
            
            new_c4 = 0
            for i in range(n):
                for j in range(i + 1, n):
                    common = np.sum(adj[i] & adj[j])
                    new_c4 += common * (common - 1) // 2
            
            if new_c4 < current_c4:
                current_c4 = new_c4
                no_improvement = 0
            else:
                adj[a, b] = 1
                adj[b, a] = 1
                adj[c, d] = 0
                adj[d, c] = 0
                no_improvement += 1
        else:
            # Disjoint edges: Delta is sum of individual deltas
            delta_rem = delta_remove_edge(a, b)
            delta_add = delta_add_edge(c, d)
            total_delta = delta_rem + delta_add
            
            if total_delta < 0:
                # Apply swap
                adj[a, b] = 0
                adj[b, a] = 0
                adj[c, d] = 1
                adj[d, c] = 1
                current_c4 += total_delta
                no_improvement = 0
            else:
                no_improvement += 1
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
    
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result
