"""Polarity graph on PG(2,5) with one extra edge, optimized via local search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Generate all 31 points in PG(2,5)
    # Points are 1D subspaces of F_5^3, represented by canonical form
    # Canonical: first non-zero coordinate is 1
    points = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero entry 1
                if a != 0:
                    pass  # a is already 1 if we normalize properly
                elif b != 0:
                    # Scale so b = 1
                    inv_b = pow(b, 3, 5)  # b^3 = b^-1 mod 5
                    a = 0
                    b = 1
                    c = (c * inv_b) % 5
                else:
                    # c != 0, scale so c = 1
                    inv_c = pow(c, 3, 5)
                    a = 0
                    b = 0
                    c = 1
                
                # Actually, let's redo this properly
                # Start over with cleaner approach
                pass
    
    # Cleaner approach: enumerate all non-zero vectors in F_5^3, 
    # identify unique 1D subspaces by normalizing
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Find first non-zero coordinate
                if a != 0:
                    # Normalize so a = 1: multiply by a^-1
                    inv_a = pow(a, 3, 5)
                    vec = (1, (b * inv_a) % 5, (c * inv_a) % 5)
                elif b != 0:
                    # a = 0, normalize so b = 1
                    inv_b = pow(b, 3, 5)
                    vec = (0, 1, (c * inv_b) % 5)
                else:
                    # a = b = 0, c != 0, normalize so c = 1
                    inv_c = pow(c, 3, 5)
                    vec = (0, 0, 1)
                
                if vec not in seen:
                    seen.add(vec)
                    points.append(vec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define the polarity: A point (x1, x2, x3) is orthogonal to (y1, y2, y3) 
    # if x1*y1 + x2*y2 + x3*y3 = 0 (standard dot product over F_5)
    # For a polarity graph, we connect i and j if points[i] · points[j] = 0
    # But we need to be careful: a point is orthogonal to itself if it's self-orthogonal
    
    # Build adjacency for the polarity graph
    # Edge (i, j) exists if points[i] · points[j] == 0 and i != j
    # Also include self-orthogonal points as loops? No, we want simple graph.
    # In polarity graphs, we typically include an edge between i and j (i≠j) 
    # if they are orthogonal, and possibly a loop if self-orthogonal.
    # For our C4 counting, we just need the simple graph edges.
    
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, x2, x3 = points[i]
            y1, y2, y3 = points[j]
            dot = (x1 * y1 + x2 * y2 + x3 * y3) % 5
            if dot == 0:
                adj[i][j] = adj[j][i] = True
                edge_set.add((i, j))
    
    num_polarity_edges = len(edge_set)
    print(f"Polarity graph has {num_polarity_edges} edges")
    
    # The polarity graph should have around 90 edges (ex(31, C4) = 90)
    # We need exactly 91 edges, so we add one more edge
    
    # Find a good extra edge to add that minimizes C4 count
    # A C4 is formed by 4 vertices a,b,c,d with edges ab, bc, cd, da
    # Adding edge (u,v) creates new C4s that include (u,v)
    # A C4 containing (u,v) has the form u-a-v-b-u or u-a-b-v-u etc.
    # More precisely: u and v share a common neighbor a, and u and v share another 
    # common neighbor b, then u-a-v-b-u is a C4 if a≠b and no other chords.
    # Actually, a C4 through edge (u,v) consists of u, v, and two other vertices a, b
    # such that a is adjacent to both u and v, b is adjacent to both u and v, 
    # and a≠b, and the cycle is u-a-v-b-u (which requires a-b not an edge for it 
    # to be an induced C4, but for subgraph C4 we just need the 4-cycle).
    
    # Wait, for counting C4 subgraphs, we count all 4-cycles as subgraphs, 
    # meaning we need 4 distinct vertices forming a cycle. The edge (u,v) plus 
    # a path of length 3 from u to v gives a C4.
    # So new C4s from adding (u,v) = number of length-3 paths from u to v 
    # that don't use edge (u,v) (which they can't since we're adding it).
    # Actually, a C4 containing edge (u,v) is determined by choosing two 
    # intermediate vertices a, b such that u-a, a-v, v-b, b-u are all edges, 
    # with a ≠ b. But we also need to ensure it's a simple cycle.
    # The number of C4s containing edge (u,v) = number of common neighbors of u and v, 
    # choose 2, but we need to be careful.
    
    # Let me think again. A 4-cycle containing edge (u,v) has the form:
    # u - a - v - b - u
    # where a and b are distinct vertices, a ≠ b, a ∉ {u,v}, b ∉ {u,v},
    # and edges (u,a), (a,v), (v,b), (b,u) all exist.
    # So for each pair of distinct common neighbors (a,b) of u and v, 
    # we get one C4: u-a-v-b-u.
    # But wait, this counts each C4 twice? No, each C4 containing (u,v) 
    # is uniquely determined by the ordered pair... actually no.
    # The cycle u-a-v-b-u is the same as u-b-v-a-u. So we should count 
    # unordered pairs {a,b} of common neighbors of u and v.
    # Number of new C4s = C(common_neighbors(u,v), 2)
    
    # But we also need to make sure that a and b are not adjacent in a way 
    # that creates a triangle... no, for subgraph counting, we just need 
    # the 4-cycle to exist as a subgraph. The presence of additional edges 
    # (like a-b) doesn't prevent it from being a C4 subgraph.
    
    # So: new_C4s(u,v) = C(|N(u) ∩ N(v)|, 2)
    
    # We want to find (u,v) not in edge_set that minimizes this.
    
    def count_common_neighbors(u, v, adj):
        count = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                count += 1
        return count
    
    # Find all non-edges and their C4 contribution
    best_extra = None
    best_c4 = float('inf')
    
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edge_set:
                continue
            cn = count_common_neighbors(u, v, adj)
            c4_contrib = cn * (cn - 1) // 2
            if c4_contrib < best_c4:
                best_c4 = c4_contrib
                best_extra = (u, v)
    
    print(f"Best extra edge: {best_extra} with {best_c4} new C4s")
    
    # Add the best extra edge
    if best_extra:
        u, v = best_extra
        adj[u][v] = adj[v][u] = True
        edge_set.add((u, v))
    
    # Now we have 91 edges. Let's count total C4s.
    # But first, let's do local search to potentially improve.
    
    def count_c4s(adj):
        """Count number of C4 subgraphs in the graph."""
        # A C4 is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da
        # We can count by: for each pair of vertices (u,v), 
        # count common neighbors, then sum C(common_neighbors, 2) over all pairs,
        # but this counts each C4 multiple times.
        # 
        # Alternative: A C4 can be counted by summing over all pairs of 
        # vertices (u,v) the number of length-2 paths from u to v, 
        # but that's not quite right either.
        #
        # Standard method: For each pair of vertices (u, v), let c(u,v) be 
        # the number of common neighbors. Then the number of C4s is 
        # (1/4) * sum over all unordered pairs {u,v} of C(c(u,v), 2)
        # Wait, that's not right either. Let me think.
        #
        # A C4 has 4 edges and 4 vertices. Each C4 contains 2 pairs of 
        # opposite vertices. For a C4 a-b-c-d-a, the opposite pairs are 
        # (a,c) and (b,d). The common neighbors of a and c are b and d 
        # (and possibly others). So C(c(a,c), 2) counts the number of 
        # ways to choose 2 common neighbors of a and c, which includes 
        # the pair (b,d) for this C4.
        #
        # So summing C(c(u,v), 2) over all unordered pairs {u,v} counts 
        # each C4 exactly twice (once for each pair of opposite vertices).
        # Therefore, total C4s = (1/2) * sum_{u<v} C(c(u,v), 2)
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                cn = count_common_neighbors(u, v, adj)
                total += cn * (cn - 1) // 2
        return total // 2
    
    current_c4 = count_c4s(adj)
    print(f"Initial C4 count: {current_c4}")
    
    best_adj = [row[:] for row in adj]
    best_c4_count = current_c4
    
    # Local search: try swapping edges
    # Remove an edge (u,v) and add a different edge (u',v')
    # This keeps the number of edges constant at 91
    
    def try_swap(adj, remove_e, add_e):
        """Try removing remove_e and adding add_e, return new C4 count."""
        u, v = remove_e
        u2, v2 = add_e
        
        # Temporarily remove (u,v)
        adj[u][v] = adj[v][u] = False
        
        # Count C4s without (u,v) but before adding (u2,v2)
        # This is expensive. Let's just compute directly.
        
        # Add (u2,v2)
        adj[u2][v2] = adj[v2][u2] = True
        
        c4 = count_c4s(adj)
        
        # Undo
        adj[u2][v2] = adj[v2][u2] = False
        adj[u][v] = adj[v][u] = True
        
        return c4
    
    # Simulated annealing / local search
    random.seed(42)
    
    current_adj = [row[:] for row in adj]
    current_c4 = current_c4_count
    
    # Precompute edge list for current graph
    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append((u, v))
        return edges
    
    current_edges = get_edge_list(current_adj)
    
    # Local search with time limit
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        # Pick a random edge to remove
        remove_e = random.choice(current_edges)
        u, v = remove_e
        
        # Pick a random non-edge to add
        # Find non-edges
        non_edges = []
        for u2 in range(n):
            for v2 in range(u2 + 1, n):
                if not current_adj[u2][v2] and (u2, v2) != remove_e:
                    non_edges.append((u2, v2))
        
        if not non_edges:
            break
        
        add_e = random.choice(non_edges)
        
        # Try the swap
        new_c4 = try_swap(current_adj, remove_e, add_e)
        
        if new_c4 < current_c4:
            # Accept the move
            current_adj[u][v] = current_adj[v][u] = False
            current_adj[add_e[0]][add_e[1]] = current_adj[add_e[1]][add_e[0]] = True
            current_edges = get_edge_list(current_adj)
            current_c4 = new_c4
            no_improve_count = 0
            
            if current_c4 < best_c4_count:
                best_c4_count = current_c4
                best_adj = [row[:] for row in current_adj]
        else:
            no_improve_count += 1
    
    print(f"Best C4 count after local search: {best_c4_count}")
    
    # Return the best adjacency as edge list
    best_edges = get_edge_list(best_adj)
    
    # Verify we have exactly 91 edges
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    
    return result
