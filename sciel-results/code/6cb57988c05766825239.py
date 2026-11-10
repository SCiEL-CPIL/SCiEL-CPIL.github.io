import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for safety
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Build edges and index mapping
    # Edge index -> (u, v)
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
                idx += 1
                
    # 2. Build squares (2-faces)
    # A square is defined by two dimensions i, j and fixed values for other dims.
    # It contains 4 edges.
    # squares: list of lists of edge indices
    squares = []
    # For each pair of dims (i, j)
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of the other d-2 bits
            # There are 2^(d-2) such squares
            other_dims = [k for k in range(d) if k != i and k != j]
            for mask in range(1 << len(other_dims)):
                # Construct the 4 vertices of the square
                # Base vertex: bits for other_dims from mask, bits i,j are 0
                v00 = 0
                for k, dim in enumerate(other_dims):
                    if mask & (1 << k):
                        v00 |= (1 << dim)
                
                v10 = v00 | (1 << i)
                v01 = v00 | (1 << j)
                v11 = v00 | (1 << i) | (1 << j)
                
                # The 4 edges of this square
                e1 = edge_to_idx[(min(v00, v10), max(v00, v10))]
                e2 = edge_to_idx[(min(v10, v11), max(v10, v11))]
                e3 = edge_to_idx[(min(v11, v01), max(v11, v01))]
                e4 = edge_to_idx[(min(v01, v00), max(v01, v00))]
                
                squares.append([e1, e2, e3, e4])
    
    n_squares = len(squares)
    
    # 3. Precompute for each edge, the squares it belongs to
    edge_to_squares = [[] for _ in range(n_edges)]
    for s_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_to_squares[e_idx].append(s_idx)
    
    # 4. Greedy Hitting Set Initialization
    # We want to find a set U of edges to remove such that every square has at least one edge in U.
    # Minimize |U|.
    
    # cover_count[s] = number of edges in U that are in square s
    cover_count = [0] * n_squares
    in_U = [False] * n_edges
    U_size = 0
    
    # Uncovered squares
    uncovered = set(range(n_squares))
    
    while uncovered:
        # Find edge that covers most uncovered squares
        best_e = -1
        max_cover = 0
        # Optimization: we can just iterate all edges, n_edges=448 is small
        for e in range(n_edges):
            if in_U[e]:
                continue
            count = 0
            for s_idx in edge_to_squares[e]:
                if s_idx in uncovered:
                    count += 1
            if count > max_cover:
                max_cover = count
                best_e = e
        
        if best_e == -1 or max_cover == 0:
            break # Should not happen if graph is connected properly, but safe
            
        # Add best_e to U
        in_U[best_e] = True
        U_size += 1
        for s_idx in edge_to_squares[best_e]:
            if s_idx in uncovered:
                cover_count[s_idx] += 1
                if cover_count[s_idx] > 0:
                    uncovered.discard(s_idx)
    
    best_U_size = U_size
    best_in_U = in_U[:]
    
    # 5. Simulated Annealing / Local Search
    # State: in_U, cover_count, U_size
    # Move: Toggle an edge
    # If removing edge (e in U): check if any square becomes uncovered.
    # If adding edge (e not in U): always valid, increases U_size.
    
    T = 1.0
    cooling = 0.999
    
    # Run until deadline
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, n_edges - 1)
        
        if in_U[e]:
            # Try to remove e
            # Check if any square in edge_to_squares[e] has cover_count == 1
            # If so, removing e makes it 0 (uncovered) -> Invalid
            valid = True
            for s_idx in edge_to_squares[e]:
                if cover_count[s_idx] == 1:
                    valid = False
                    break
            
            if valid:
                # Remove e
                in_U[e] = False
                U_size -= 1
                for s_idx in edge_to_squares[e]:
                    cover_count[s_idx] -= 1
                
                if U_size < best_U_size:
                    best_U_size = U_size
                    best_in_U = in_U[:]
            else:
                # Invalid removal, skip
                pass
        else:
            # Try to add e
            # This increases U_size, so it's a "worsening" move
            # Accept with probability exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                in_U[e] = True
                U_size += 1
                for s_idx in edge_to_squares[e]:
                    cover_count[s_idx] += 1
                # Note: We don't update best_U_size here as it got worse
                # But it might allow future removals
    
        T *= cooling
        if T < 0.01:
            T = 0.01 # Keep some exploration

    # 6. Construct output
    # The kept edges are those NOT in best_in_U
    out = []
    for i in range(n_edges):
        if not best_in_U[i]:
            u, v = edges[i]
            out.append([u, v])
            
    return out
