import random
import math
from collections import defaultdict

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_triangles(n, adj):
    """Find all triangles in the graph."""
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        triangles.append((u, v, w))
    return triangles

def _max_triangle_degree_sum(n, adj):
    """Compute the maximum degree sum over all triangles."""
    if n <= 2:
        return 0
    triangles = _find_triangles(n, adj)
    if not triangles:
        return 0
    max_sum = 0
    for (u, v, w) in triangles:
        s = len(adj[u]) + len(adj[v]) + len(adj[w])
        if s > max_sum:
            max_sum = s
    return max_sum

def construct():
    n = 16
    target_edges = 65
    
    # Structure: Bipartite graph between A (0-7) and B (8-15) with 64 edges
    # Plus one intra-part edge.
    # We will optimize the bipartite edges and the intra-part edge location.
    
    def initialize():
        # Create a random bipartite graph with 64 edges
        # Each vertex in A connects to 8 vertices in B on average?
        # To have 64 edges total, and 8+8 vertices, average degree is 8.
        # Let's just pick 64 random pairs from the 64 possible A-B pairs.
        # There are 8*8=64 possible pairs. So we must pick ALL of them?
        # No, 8*8 = 64. So a complete bipartite graph K_{8,8} has exactly 64 edges.
        # If we start with K_{8,8}, we have no freedom in the bipartite part if we want 64 edges.
        # Wait, the problem says 65 edges.
        # If we use K_{8,8} (64 edges) + 1 intra edge, we have 65 edges.
        # But K_{8,8} is fixed. The only choice is which intra edge to add.
        # In K_{8,8}, every vertex has degree 8.
        # If we add edge (u,v) in A:
        # deg(u) = 9, deg(v) = 9.
        # Common neighbors of u and v in B: All 8 vertices in B (since it's complete bipartite).
        # For any w in B, deg(w) = 8.
        # Triangle (u,v,w) sum = 9 + 9 + 8 = 26.
        # This is the baseline.
        
        # Can we do better?
        # We need 64 bipartite edges. K_{8,8} is the only 8-regular bipartite graph on 8+8 vertices?
        # No, we can have non-regular bipartite graphs with 64 edges.
        # Example: Some vertices in A have degree 9, some 7.
        # But sum of degrees in A must be 64.
        # If we make the bipartite graph non-regular, we might reduce the max triangle sum.
        
        # Let's represent the bipartite graph by a matrix M[8][8] of 0/1.
        # Sum of M is 64.
        # Intra edge (i, j) in A.
        # Triangles are (i, j, k) for k in B such that M[i][k]=1 and M[j][k]=1.
        # Sum = (row_sum(i) + 1) + (row_sum(j) + 1) + col_sum(k).
        # We want to minimize max over all such triangles.
        
        # Initialization: Random regular or near-regular bipartite graph.
        # Since sum=64 and 8 vertices, average row sum is 8.
        # Let's start with all 1s (K_{8,8}) and then perform swaps to unbalance degrees
        # strategically to lower the max sum.
        
        # Actually, if we unbalance, say row_sum(i) becomes 7, then row_sum(k) must become 9 somewhere else.
        # If the intra edge is between two low-degree vertices, their contribution is lower.
        # But their common neighbors might have higher column sums?
        
        # Let's try a local search on the bipartite matrix and intra edge.
        
        # State: 
        # - mat: 8x8 binary matrix
        # - intra_u, intra_v: indices in A (0-7)
        
        return None # Placeholder, logic in main loop

    # We will use a specific local search on the bipartite structure.
    # State: mat (list of lists), intra_u, intra_v
    
    def init_state():
        # Start with K_{8,8}
        mat = [[1]*8 for _ in range(8)]
        # Random intra edge
        intra_u, intra_v = random.sample(range(8), 2)
        return mat, intra_u, intra_v

    def get_score(mat, intra_u, intra_v):
        # Calculate row sums and col sums
        row_sums = [sum(mat[i]) for i in range(8)]
        col_sums = [sum(mat[i][j] for i in range(8)) for j in range(8)]
        
        max_sum = 0
        # Triangles are (intra_u, intra_v, k) for k in 0..7
        # such that mat[intra_u][k] == 1 and mat[intra_v][k] == 1
        for k in range(8):
            if mat[intra_u][k] == 1 and mat[intra_v][k] == 1:
                # deg(intra_u) = row_sums[intra_u] + 1
                # deg(intra_v) = row_sums[intra_v] + 1
                # deg(k) = col_sums[k]
                s = (row_sums[intra_u] + 1) + (row_sums[intra_v] + 1) + col_sums[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    # Local Search
    # Moves:
    # 1. Swap edges in bipartite graph: change mat[i][j] from 1 to 0 and mat[i'][j'] from 0 to 1
    #    This keeps total edges at 64.
    # 2. Change intra edge: pick new intra_u, intra_v.
    
    mat, intra_u, intra_v = init_state()
    current_score = get_score(mat, intra_u, intra_v)
    best_score = current_score
    best_mat = [row[:] for row in mat]
    best_intra_u, best_intra_v = intra_u, intra_v
    
    temp = 10.0
    cooling = 0.999
    min_temp = 0.01
    max_iters = 50000
    
    for it in range(max_iters):
        if temp < min_temp:
            break
            
        new_mat = [row[:] for row in mat]
        new_intra_u, new_intra_v = intra_u, intra_v
        
        # Choose move type
        if random.random() < 0.5:
            # Swap bipartite edges
            # Pick random 1 to remove
            ones = [(i,j) for i in range(8) for j in range(8) if new_mat[i][j]==1]
            zeros = [(i,j) for i in range(8) for j in range(8) if new_mat[i][j]==0]
            if not ones or not zeros:
                continue
            i1, j1 = random.choice(ones)
            i2, j2 = random.choice(zeros)
            new_mat[i1][j1] = 0
            new_mat[i2][j2] = 1
        else:
            # Change intra edge
            new_intra_u, new_intra_v = random.sample(range(8), 2)
            
        new_score = get_score(new_mat, new_intra_u, new_intra_v)
        
        if new_score <= current_score:
            mat = new_mat
            intra_u, intra_v = new_intra_u, new_intra_v
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_mat = [row[:] for row in mat]
                best_intra_u, best_intra_v = intra_u, intra_v
        else:
            delta = new_score - current_score
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                mat = new_mat
                intra_u, intra_v = new_intra_u, new_intra_v
                current_score = new_score
                
        temp *= cooling
        
    # Reconstruct edges from best state
    edges = []
    # Bipartite edges
    for i in range(8):
        for j in range(8):
            if best_mat[i][j] == 1:
                edges.append([i, 8+j])
                
    # Intra edge
    edges.append([best_intra_u, best_intra_v])
    
    return {'n': n, 'edges': edges}
