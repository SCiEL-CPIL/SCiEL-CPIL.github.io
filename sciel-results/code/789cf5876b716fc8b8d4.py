import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Define Grötzsch Graph (Mycielskian of C5)
    # Vertices 0..10
    groetzsch_edges = []
    # Cycle 0-1-2-3-4-0
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    groetzsch_edges.extend(cycle_edges)
    
    # Mycielski construction
    # Copies 5..9 correspond to 0..4
    # Edge (v', u) exists if (v, u) exists in original
    # v' = 5 + v
    # Neighbors in C5:
    # 0: 1, 4
    # 1: 0, 2
    # 2: 1, 3
    # 3: 2, 4
    # 4: 3, 0
    
    neighbors = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 0]
    }
    
    for v in range(5):
        v_prime = 5 + v
        for u in neighbors[v]:
            groetzsch_edges.append((v_prime, u))
            
    # Root 10 connected to all copies 5..9
    for v in range(5):
        groetzsch_edges.append((10, 5 + v))
        
    # 2. Search for optimal blow-up sizes
    # Total vertices = 16
    # Sizes s[0..10] sum to 16
    # We want to maximize edges in the blow-up.
    # Edges in blow-up = sum_{(i,j) in E(G)} s_i * s_j
    
    # Generate all compositions of 16 into 11 parts
    def generate_compositions(n, k):
        if k == 1:
            yield [n]
        else:
            for i in range(1, n - k + 2):
                for tail in generate_compositions(n - i, k - 1):
                    yield [i] + tail
                    
    best_edges_count = -1
    best_sizes = None
    
    # Iterate over all compositions of 16 into 11 parts
    # Due to symmetry:
    # - Vertices 0-4 are symmetric (cycle vertices)
    # - Vertices 5-9 are symmetric (copy vertices)
    # - Vertex 10 is unique (root)
    # So we sort sizes descending and assign:
    # - Largest to 10
    # - Next 5 to 5-9
    # - Last 5 to 0-4
    
    for comp in generate_compositions(16, 11):
        comp_sorted = sorted(comp, reverse=True)
        
        s = [0] * 11
        s[10] = comp_sorted[0]
        for i in range(5):
            s[5+i] = comp_sorted[1+i]
        for i in range(5):
            s[i] = comp_sorted[6+i]
            
        edge_count = 0
        for u, v in groetzsch_edges:
            edge_count += s[u] * s[v]
            
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_sizes = s[:]
            
    # 3. Construct the 16-vertex graph from best_sizes
    adj_matrix = [[False] * 16 for _ in range(16)]
    offsets = [0] * 12
    for i in range(11):
        offsets[i+1] = offsets[i] + best_sizes[i]
        
    for u, v in groetzsch_edges:
        for i in range(best_sizes[u]):
            u_node = offsets[u] + i
            for j in range(best_sizes[v]):
                v_node = offsets[v] + j
                adj_matrix[u_node][v_node] = True
                adj_matrix[v_node][u_node] = True
                
    # 4. Greedy Edge Addition with multiple restarts
    # Try to add any edge (u, v) if it doesn't create a triangle
    # and the graph remains non-3-colorable (we assume it does since we start non-3-colorable)
    
    def count_edges(adj):
        count = 0
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u][v]:
                    count += 1
        return count
    
    def greedy_add_edges(adj, deadline):
        """Greedy addition of triangle-safe edges."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle order to break symmetry
            pairs = [(u, v) for u in range(16) for v in range(u + 1, 16) if not adj[u][v]]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() >= deadline:
                    break
                if adj[u][v]:
                    continue
                # Check if adding (u, v) creates a triangle
                creates_triangle = False
                for w in range(16):
                    if adj[u][w] and adj[w][v]:
                        creates_triangle = True
                        break
                if not creates_triangle:
                    adj[u][v] = True
                    adj[v][u] = True
                    changed = True
        return adj
    
    # Try multiple restarts with different random seeds
    best_final_edges = -1
    best_final_adj = None
    
    # First, try deterministic greedy
    adj_copy = [row[:] for row in adj_matrix]
    adj_copy = greedy_add_edges(adj_copy, deadline)
    ec = count_edges(adj_copy)
    if ec > best_final_edges:
        best_final_edges = ec
        best_final_adj = [row[:] for row in adj_copy]
    
    # Now try randomized restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_restarts = int(remaining_time / 2)  # ~2s per restart
        for restart in range(num_restarts):
            if time.time() >= deadline - 1:
                break
            random.seed(restart)
            adj_copy = [row[:] for row in adj_matrix]
            adj_copy = greedy_add_edges(adj_copy, deadline)
            ec = count_edges(adj_copy)
            if ec > best_final_edges:
                best_final_edges = ec
                best_final_adj = [row[:] for row in adj_copy]
                
    # 5. Return edges from the best graph found
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if best_final_adj[u][v]:
                edges.append([u, v])
                
    return edges

# Set a default seed for reproducibility if needed, but random.seed is called per restart
random.seed(42)
