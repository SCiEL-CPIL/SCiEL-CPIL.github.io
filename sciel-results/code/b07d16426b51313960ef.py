import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit
    
    # 1. Generate the best Grötzsch blow-up (47 edges)
    # Grötzsch Graph Edges (11 vertices)
    groetzsch_edges = []
    # Cycle 0-1-2-3-4-0
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    groetzsch_edges.extend(cycle_edges)
    
    # Mycielski construction
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
        
    # Find optimal sizes for 16 vertices
    # Re-use the logic from Inspiration 1 to find the specific sizes
    def generate_compositions(n, k):
        if k == 1:
            yield [n]
        else:
            for i in range(1, n - k + 2):
                for tail in generate_compositions(n - i, k - 1):
                    yield [i] + tail
                    
    best_edges_count = -1
    best_sizes = None
    
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
            
    # Construct the 16-vertex graph
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
                
    # 2. Greedy Edge Addition
    # Try to add any edge (u, v) if it doesn't create a triangle
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(16):
            for v in range(u + 1, 16):
                if not adj_matrix[u][v]:
                    # Check if adding (u, v) creates a triangle
                    creates_triangle = False
                    for w in range(16):
                        if adj_matrix[u][w] and adj_matrix[w][v]:
                            creates_triangle = True
                            break
                    if not creates_triangle:
                        adj_matrix[u][v] = True
                        adj_matrix[v][u] = True
                        changed = True
                        
    # 3. Verify 3-colorability (should be False, i.e., not 3-colorable)
    # We start with a non-3-colorable graph, and adding edges preserves this property.
    # So we are guaranteed chi >= 4.
    
    # 4. Count edges
    edge_count = 0
    for u in range(16):
        for v in range(u + 1, 16):
            if adj_matrix[u][v]:
                edge_count += 1
                
    # 5. Return edges
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if adj_matrix[u][v]:
                edges.append([u, v])
                
    return edges
