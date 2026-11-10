import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit (safe margin before 150s kill)
    
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
    
    # Generate candidate size vectors
    # We can use a recursive generator or just iterate over combinations of increments.
    # Base sizes: [1]*11
    # We have 5 increments to distribute.
    # This is a "stars and bars" problem with 5 stars and 11 bars.
    # Number of solutions: C(5+11-1, 11-1) = C(15, 10) = C(15, 5) = 3003.
    # This is very small! We can iterate over ALL of them.
    
    def generate_compositions(n, k):
        # Generate compositions of n into k parts (positive integers)
        # Here we start with 1s, so we distribute n-k extra units.
        # Actually, let's just generate all s_i >= 1 summing to 16.
        # Recursion is fine.
        if k == 1:
            yield [n]
        else:
            for i in range(1, n - k + 2): # i from 1 to n-k+1
                for tail in generate_compositions(n - i, k - 1):
                    yield [i] + tail
                    
    # We want to assign sizes to specific vertices to maximize edges.
    # The objective function depends on WHICH vertex gets WHICH size.
    # So we need to permute the sizes among the 11 vertices.
    # 11! is too large.
    # However, the graph has symmetry.
    # Vertices 0-4 are symmetric.
    # Vertices 5-9 are symmetric.
    # Vertex 10 is unique.
    # So we only need to choose:
    # s_root (size of 10)
    # s_copy (size of 5-9) -- but they can be different?
    # Due to symmetry of C5, we can assume 0-4 have sizes a,b,c,d,e and 5-9 have sizes f,g,h,i,j.
    # But to maximize, we should probably make symmetric vertices have same sizes?
    # Not necessarily, but it's a good starting point.
    # Let's just iterate over all compositions of 16 into 11 parts, and for each,
    # assign them to vertices in a way that maximizes edges.
    # Since we can't permute 11! times, we'll rely on the fact that high degree vertices should get high sizes.
    # We will sort the sizes in descending order and assign them to vertices sorted by degree descending?
    # Degrees: 10 (deg 5), 5-9 (deg 3), 0-4 (deg 2).
    # So largest size -> 10.
    # Next 5 largest -> 5-9.
    # Smallest 5 -> 0-4.
    
    # Let's refine:
    # 1. Generate all compositions of 16 into 11 parts.
    # 2. For each composition, sort sizes descending.
    # 3. Assign largest to 10.
    # 4. Assign next 5 to 5-9 (any order, they are symmetric).
    # 5. Assign last 5 to 0-4 (any order, they are symmetric).
    # 6. Calculate edge count.
    # 7. Keep the best one.
    
    # This covers the "symmetric" optimal solution.
    
    best_edges_count = -1
    best_sizes = None
    
    for comp in generate_compositions(16, 11):
        # Sort descending
        comp_sorted = sorted(comp, reverse=True)
        
        # Assign
        s = [0] * 11
        s[10] = comp_sorted[0]
        
        # Assign 5-9
        for i in range(5):
            s[5+i] = comp_sorted[1+i]
            
        # Assign 0-4
        for i in range(5):
            s[i] = comp_sorted[6+i]
            
        # Calculate edges
        edge_count = 0
        for u, v in groetzsch_edges:
            edge_count += s[u] * s[v]
            
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_sizes = s[:]
            
    # 3. Construct the final graph from best_sizes
    # We will use an adjacency matrix for efficient edge addition checks
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
                
    # 4. Greedy Edge Addition
    # Try to add any edge (u, v) if it doesn't create a triangle
    # This is a local search to maximize edges while maintaining triangle-free property
    # Since the graph is already not 3-colorable (blow-up of Grötzsch), adding edges preserves this property
    
    # Optimization: Precompute common neighbors?
    # For n=16, simple check is O(n) per edge, O(n^3) total per pass.
    # 16^3 = 4096 operations per pass. Very fast.
    
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
                        
    # 5. Verification (Optional but good for safety)
    # Check triangle-free and 4-chromatic?
    # Given the construction from Grötzsch, it is guaranteed.
    # But let's ensure no duplicates and valid range.
    # Vertices are 0..15.
    
    # 6. Count edges and return
    final_edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if adj_matrix[u][v]:
                final_edges.append([u, v])
                
    return final_edges

# Helper for compositions
def generate_compositions(n, k):
    if k == 1:
        yield [n]
    else:
        for i in range(1, n - k + 2):
            for tail in generate_compositions(n - i, k - 1):
                yield [i] + tail
