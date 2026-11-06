import random
import time
import sys

N = 18
ALL_MASK = (1 << N) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def is_k4_free(adj):
        # Check if graph is K4-free
        # A K4 is a set of 4 vertices all connected to each other.
        # We can check for every pair (u, v) if they share a common neighbor w, 
        # and if that w shares a common neighbor x with u and v such that w-x is an edge.
        # Simpler: For each triangle (u, v, w), check if any of u, v, w is connected to a 4th vertex 
        # that is also connected to the other two.
        
        # Efficient check:
        # For each pair (u, v), let C = adj[u] & adj[v].
        # If C contains an edge (w, x), then u, v, w, x form a K4.
        # So for each u, v, check if subgraph induced by C has any edge.
        
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    # Check if any w in C is connected to any x in C
                    # C is a bitmask.
                    # We can iterate through bits of C
                    c_list = []
                    c = C
                    while c:
                        lsb = c & -c
                        idx = lsb.bit_length() - 1
                        c_list.append(idx)
                        c -= lsb
                    
                    for i in range(len(c_list)):
                        w = c_list[i]
                        # Check if w is connected to any other in c_list
                        # adj[w] & C (excluding w itself)
                        if adj[w] & (C ^ (1 << w)):
                            return False
        return True

    def get_heuristic_tf_size(adj):
        # Greedy removal of vertices to eliminate all triangles
        # Returns the size of the resulting triangle-free subgraph
        remaining = ALL_MASK
        # Precompute triangles for current graph? 
        # Or just find one triangle at a time.
        
        # To speed up, we can use a counter for triangles or just search.
        # Search for a triangle:
        # For each u, for each v in adj[u], for each w in adj[v] & adj[u] & (1<<w > 1<<v)
        
        while True:
            # Find a triangle
            tri_found = False
            # Iterate u
            for u in range(N):
                if not (remaining & (1 << u)): continue
                # Iterate v > u
                v_mask = remaining & adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    lsb_v = v_mask & -v_mask
                    v = lsb_v.bit_length() - 1
                    # w > v, w in adj[u] & adj[v]
                    w_mask = remaining & adj[u] & adj[v] & (~((1 << (v + 1)) - 1))
                    if w_mask:
                        # Found triangle (u, v, w)
                        tri_found = True
                        # Remove one vertex, say u
                        remaining ^= (1 << u)
                        break
                    v_mask ^= lsb_v
                if tri_found:
                    break
            if not tri_found:
                break
        
        return bin(remaining).count('1')

    def exact_tf_size(adj):
        # Exact calculation for final verification
        # Use branch and bound or simple iteration if fast enough
        # For n=18, we can use a recursive solver
        # But for now, let's assume the heuristic is good enough for ranking,
        # and we might refine the best candidate.
        
        # Actually, let's implement a simple exact solver using recursion
        # State: set of vertices, set of triangles
        # This might be slow. Let's stick to heuristic for search, 
        # and maybe do a local improvement on the best found.
        
        pass

    def generate_random_k4_free():
        adj = [0] * N
        # Start with empty graph
        # Add edges randomly
        edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(edges)
        for u, v in edges:
            # Check if adding (u, v) creates a K4
            # Common neighbors C = adj[u] & adj[v]
            C = adj[u] & adj[v]
            # Check if C has an edge
            has_edge = False
            c_list = []
            c = C
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                c_list.append(idx)
                c -= lsb
            for i in range(len(c_list)):
                w = c_list[i]
                if adj[w] & (C ^ (1 << w)):
                    has_edge = True
                    break
            if not has_edge:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_adj = None
    best_score = N + 1
    
    # Run local search
    current_adj = generate_random_k4_free()
    current_score = get_heuristic_tf_size(current_adj)
    
    if current_score < best_score:
        best_score = current_score
        best_adj = [x for x in current_adj]
        
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    steps = 0
    while time.time() < deadline:
        if T < T_min:
            # Restart
            T = 1.0
            current_adj = generate_random_k4_free()
            current_score = get_heuristic_tf_size(current_adj)
            if current_score < best_score:
                best_score = current_score
                best_adj = [x for x in current_adj]
            
        # Pick a random edge to flip
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        # Try flipping edge (u, v)
        old_adj = [x for x in current_adj]
        
        if current_adj[u] & (1 << v):
            # Remove edge
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
        else:
            # Add edge, check K4
            C = current_adj[u] & current_adj[v]
            has_edge = False
            c_list = []
            c = C
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                c_list.append(idx)
                c -= lsb
            for i in range(len(c_list)):
                w = c_list[i]
                if current_adj[w] & (C ^ (1 << w)):
                    has_edge = True
                    break
            if has_edge:
                # Cannot add
                current_adj = old_adj
                continue
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        new_score = get_heuristic_tf_size(current_adj)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [x for x in current_adj]
        else:
            delta = new_score - current_score
            if random.random() < pow(2.71828, -delta / T):
                current_score = new_score
            else:
                current_adj = old_adj
        
        T *= cooling
        steps += 1
        
    # Convert best_adj to list of edges
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges

# Run once to verify
if __name__ == "__main__":
    e = construct()
    print(len(e))
