import random
import time
import sys

N = 18
DEADLINE = 115.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Bitmask utilities
    def bit_count(x):
        return bin(x).count('1')
        
    def get_vertices(mask):
        verts = []
        for i in range(N):
            if mask & (1 << i):
                verts.append(i)
        return verts

    # Evaluation: Largest Triangle-Free Induced Subgraph
    # Uses Branch and Bound
    def largest_triangle_free(adj_masks):
        best = 0
        # Order vertices by degree descending to prune earlier? 
        # Or just standard order.
        # We want to find a large independent set in the "triangle graph" ? No.
        # We want a subset S such that no triangle in G is contained in S.
        # This is equivalent to finding a maximum subset of vertices that does not contain any triangle of G.
        
        # Precompute triangles as bitmasks
        triangles = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj_masks[i] & (1 << j):
                    common = adj_masks[i] & adj_masks[j]
                    # Remove i and j from common? No, common contains neighbors of i and j.
                    # A triangle (i, j, k) requires k in common and k != i, j.
                    # Since i, j are not neighbors of themselves in simple graph, 
                    # and we check k > j to avoid duplicates.
                    k_mask = common & ~((1 << i) | (1 << j))
                    while k_mask:
                        lsb = k_mask & -k_mask
                        k = lsb.bit_length() - 1
                        if k > j:
                            triangles.append((1 << i) | (1 << j) | (1 << k))
                        k_mask ^= lsb
                        
        # If no triangles, the whole graph is triangle-free
        if not triangles:
            return N

        # Branch and Bound
        # State: remaining vertices mask, current selected mask
        # Pruning: 
        # 1. If current_size + bit_count(remaining) <= best: return
        # 2. If current selected mask contains a triangle: return (invalid path)
        
        # To check if current_mask contains a triangle quickly:
        # We can maintain a 'forbidden' set? 
        # Simpler: Just check triangles against current_mask.
        # Since N=18, triangles count is small (< 100 usually).
        
        def bnb(remaining, current_mask, current_size):
            nonlocal best
            
            if current_size + bit_count(remaining) <= best:
                return
            
            # Check if current_mask is triangle-free
            # Optimization: Only check if current_size >= 3
            if current_size >= 3:
                for tri in triangles:
                    if (current_mask & tri) == tri:
                        return
            
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Heuristic: Pick a vertex from remaining
            # Try including it first (to find large sets quickly)
            # Choose vertex with highest degree in remaining? Or just lowest index?
            # Let's pick the vertex with the most neighbors in remaining to prune "triangle risks" earlier?
            # Actually, for independent set in complement... 
            # Let's just iterate.
            
            # Pick a vertex v from remaining
            # Heuristic: Pick v such that it is in the most triangles within remaining?
            # Simpler: Pick v with max degree in remaining to reduce branching factor?
            # Let's try picking the vertex with the highest index to be deterministic and fast?
            # Or random? No, B&B needs determinism for correctness of bound? No, just search order.
            
            # Let's pick the vertex v in remaining that is part of the most triangles 
            # that are subsets of (current_mask U remaining).
            # This is expensive. 
            # Standard heuristic: Pick v with max degree in remaining.
            
            v = -1
            max_deg = -1
            temp_rem = remaining
            while temp_rem:
                lsb = temp_rem & -temp_rem
                u = lsb.bit_length() - 1
                deg = bit_count(adj_masks[u] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    v = u
                temp_rem ^= lsb
            
            if v == -1:
                return

            # Option 1: Include v
            new_mask = current_mask | (1 << v)
            # Check if adding v creates a triangle with current_mask
            # We already check at start of bnb, but we can prune earlier.
            # If v forms a triangle with any two vertices in current_mask, we cannot include v.
            # Triangle (v, a, b) where a,b in current_mask.
            # This happens if (adj_masks[v] & current_mask) has an edge (a,b).
            # i.e., if there exists a in (adj_masks[v] & current_mask) such that adj_masks[a] & (adj_masks[v] & current_mask) != 0
            
            neigh_in_curr = adj_masks[v] & current_mask
            # Check if neigh_in_curr has any edge
            valid_include = True
            temp_neigh = neigh_in_curr
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                u = lsb.bit_length() - 1
                if adj_masks[u] & neigh_in_curr:
                    valid_include = False
                    break
                temp_neigh ^= lsb
            
            if valid_include:
                bnb(remaining ^ (1 << v), new_mask, current_size + 1)
            
            # Option 2: Exclude v
            bnb(remaining ^ (1 << v), current_mask, current_size)

        bnb((1 << N) - 1, 0, 0)
        return best

    def is_k4_free(adj_masks):
        # Check if adding any edge would create K4? No, check current graph.
        # A K4 is a set of 4 vertices with all 6 edges.
        # Iterate over all triangles (i, j, k).
        # For each triangle, check if there is a vertex l connected to i, j, k.
        # If so, {i, j, k, l} is a K4.
        
        # Precompute triangles
        triangles = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj_masks[i] & (1 << j):
                    common = adj_masks[i] & adj_masks[j]
                    k_mask = common & ~((1 << i) | (1 << j))
                    while k_mask:
                        lsb = k_mask & -k_mask
                        k = lsb.bit_length() - 1
                        if k > j:
                            triangles.append((i, j, k))
                        k_mask ^= lsb
        
        for i, j, k in triangles:
            # Common neighbors of i, j, k
            common = adj_masks[i] & adj_masks[j] & adj_masks[k]
            # Remove i, j, k
            common &= ~((1 << i) | (1 << j) | (1 << k))
            if common != 0:
                return False
        return True

    def can_add_edge(adj_masks, u, v):
        if adj_masks[u] & (1 << v):
            return False
        # Check if adding (u, v) creates a K4.
        # This happens if there are two vertices w1, w2 in N(u) ∩ N(v) such that (w1, w2) is an edge.
        common = adj_masks[u] & adj_masks[v]
        common &= ~((1 << u) | (1 << v))
        
        # Check if induced subgraph on common contains an edge
        temp = common
        while temp:
            lsb = temp & -temp
            w1 = lsb.bit_length() - 1
            # Check if w1 has any neighbor in common
            if adj_masks[w1] & common:
                return False
            temp ^= lsb
        return True

    def get_largest_tf_set(adj_masks):
        """Returns a bitmask of a largest triangle-free induced subgraph."""
        best_mask = 0
        best_size = 0
        
        # Reuse logic from largest_triangle_free but track mask
        triangles = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj_masks[i] & (1 << j):
                    common = adj_masks[i] & adj_masks[j]
                    k_mask = common & ~((1 << i) | (1 << j))
                    while k_mask:
                        lsb = k_mask & -k_mask
                        k = lsb.bit_length() - 1
                        if k > j:
                            triangles.append((1 << i) | (1 << j) | (1 << k))
                        k_mask ^= lsb
                        
        if not triangles:
            return (1 << N) - 1

        def bnb(remaining, current_mask, current_size):
            nonlocal best_mask, best_size
            
            if current_size + bit_count(remaining) <= best_size:
                return
            
            if current_size >= 3:
                for tri in triangles:
                    if (current_mask & tri) == tri:
                        return
            
            if remaining == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            # Heuristic: Pick v with max degree in remaining
            v = -1
            max_deg = -1
            temp_rem = remaining
            while temp_rem:
                lsb = temp_rem & -temp_rem
                u = lsb.bit_length() - 1
                deg = bit_count(adj_masks[u] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    v = u
                temp_rem ^= lsb
            
            if v == -1:
                return

            # Option 1: Include v
            neigh_in_curr = adj_masks[v] & current_mask
            valid_include = True
            temp_neigh = neigh_in_curr
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                u = lsb.bit_length() - 1
                if adj_masks[u] & neigh_in_curr:
                    valid_include = False
                    break
                temp_neigh ^= lsb
            
            if valid_include:
                bnb(remaining ^ (1 << v), current_mask | (1 << v), current_size + 1)
            
            # Option 2: Exclude v
            bnb(remaining ^ (1 << v), current_mask, current_size)

        bnb((1 << N) - 1, 0, 0)
        return best_mask

    def adj_masks_to_edges(adj_masks):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj_masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def random_k4_free_graph():
        """Generate a random K4-free graph by greedily adding edges."""
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_score = N
    best_adj = [0] * N
    
    # Main Loop
    while time.time() - start_time < DEADLINE:
        # Random restart / Initialization
        adj = random_k4_free_graph()
        
        current_score = largest_triangle_free(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
            
        # Local Search
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            
            # Find the specific largest triangle-free set
            tf_mask = get_largest_tf_set(adj)
            tf_vertices = get_vertices(tf_mask)
            
            if len(tf_vertices) < 2:
                break
            
            # Try to add an edge within tf_vertices
            # Shuffling to break symmetry
            random.shuffle(tf_vertices)
            
            move_made = False
            for i in range(len(tf_vertices)):
                for j in range(i + 1, len(tf_vertices)):
                    u = tf_vertices[i]
                    v = tf_vertices[j]
                    
                    # Check if edge exists
                    if adj[u] & (1 << v):
                        continue
                    
                    # Check if we can add it (K4-free)
                    if not can_add_edge(adj, u, v):
                        continue
                    
                    # Tentatively add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    new_score = largest_triangle_free(adj)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        move_made = True
                        break
                    else:
                        # Revert
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                
                if improved:
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj[:]
                    break
            
            if not improved:
                # Try random perturbation to escape local minimum
                # Remove a random edge or add a random edge outside TF set?
                # Let's try adding a random edge anywhere that is K4-free
                all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
                random.shuffle(all_edges)
                perturbed = False
                for u, v in all_edges:
                    if not (adj[u] & (1 << v)) and can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        new_score = largest_triangle_free(adj)
                        if new_score <= current_score:
                            current_score = new_score
                            perturbed = True
                            if new_score < best_score:
                                best_score = new_score
                                best_adj = adj[:]
                            break
                        else:
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                
                if not perturbed:
                    break

    return adj_masks_to_edges(best_adj)

# Entry point
if __name__ == "__main__":
    construct()
