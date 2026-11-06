import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n = 31
    target_edges = 91
    
    def get_points(q):
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        return points

    def build_polarity_graph(points, q, a, b, c, d, e, f):
        """
        Build graph based on symmetric bilinear form:
        a*x1*x2 + b*(x1*y2 + x2*y1) + c*(x1*z2 + x2*z1) + d*y1*y2 + e*(y1*z2 + y2*z1) + f*z1*z2 = 0
        """
        adj = np.zeros((n, n), dtype=np.int8)
        edges = []
        
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                val = (a * p1[0] * p2[0] + 
                       b * (p1[0] * p2[1] + p1[1] * p2[0]) + 
                       c * (p1[0] * p2[2] + p1[2] * p2[0]) + 
                       d * p1[1] * p2[1] + 
                       e * (p1[1] * p2[2] + p1[2] * p2[1]) + 
                       f * p1[2] * p2[2]) % q
                if val == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    edges.append((i, j))
        return adj, edges

    def count_c4s(adj, edges):
        # Count C4s in the graph
        # A C4 is a set of 4 vertices {u,v,w,x} with edges (u,v), (v,w), (w,x), (x,u)
        # We can count by iterating over pairs of edges?
        # Or use the fact that C4 count = sum over all pairs of vertices (u,v) of (common neighbors choose 2)?
        # No, that counts K4 and C4s sharing edges?
        # Number of C4s = (1/8) * sum_{u,v} (d(u,v,2) - d(u,v,1))^2 ? No.
        # Simpler: For each pair of edges (u,v) and (w,x) that are disjoint?
        # Let's use the standard formula:
        # C4s = (1/2) * sum_{u < v} ( (A^2)_{uv} - A_{uv} ) * ( (A^2)_{uv} - A_{uv} ) / 2 ?
        # Actually, number of C4s = (1/8) * trace(A^4) - ...
        # Let's just use a direct count for n=31.
        # Iterate over all 4-tuples? 31^4 is ~ 1M. Fast enough.
        count = 0
        # To avoid double counting, we can count sets {u,v,w,x}
        # Or count paths of length 2 between u and v?
        # If u and v have k common neighbors, they form k * (k-1) / 2 C4s?
        # No, each pair of common neighbors (w1, w2) forms a C4 u-w1-v-w2-u.
        # So for each pair (u,v), let k be the number of common neighbors.
        # The number of C4s containing edge (u,v) is not defined if (u,v) is not an edge?
        # A C4 is a cycle. It has 4 edges.
        # If we sum over all pairs of vertices (u,v) the number of common neighbors, that's not it.
        # Number of C4s = (1/2) * sum_{u,v} ( (A^2)_{uv} choose 2 )?
        # Let's check.
        # (A^2)_{uv} is the number of paths of length 2 between u and v.
        # If u and v are adjacent, a path of length 2 u-w-v forms a triangle.
        # If u and v are not adjacent, a path of length 2 u-w-v is part of a C4?
        # No, a C4 is u-v-w-x-u.
        # The pairs of opposite vertices are (u,w) and (v,x).
        # So for each pair of opposite vertices (u,w), the number of C4s where u,w are opposite is (common neighbors of u and w choose 2).
        # Summing over all pairs (u,w) and dividing by 2 (since each C4 has 2 pairs of opposite vertices) gives the total C4s.
        # Formula: C4s = (1/2) * sum_{u < w} ( (A^2)_{uw} choose 2 )
        
        A2 = np.dot(adj, adj)
        # A2[u,v] is number of length-2 paths.
        # We need (A2[u,v] choose 2) for all u != v.
        # Note: If u,v are adjacent, A2[u,v] includes paths via neighbors.
        # The formula holds for all pairs u,v.
        
        total = 0
        for u in range(n):
            for w in range(u + 1, n):
                k = A2[u, w]
                total += k * (k - 1) // 2
        return total // 2

    def min_cost_edge(adj, edges, edge_set):
        """Find the non-edge with minimum length-3 paths."""
        best_cost = float('inf')
        best_pair = None
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            
            # Count length-3 paths: sum(adj[u, x] * adj[v, y] * adj[x, y] for x,y)
            # Vectorized:
            count = np.sum(adj[u][:, None] * adj[v][None, :] * adj)
            
            if count < best_cost:
                best_cost = count
                best_pair = (u, v)
                
        return best_pair, best_cost

    q = 5
    points = get_points(q)
    
    best_total_c4 = float('inf')
    best_edges_final = None
    
    # Try a few variations of the polarity
    # Standard: x1*x2 + y1*y2 - z1*z2 = 0
    # This corresponds to a=1, d=1, f=-1 (which is 4 mod 5)
    # Others:
    # a=1, d=1, f=1 (x1x2 + y1y2 + z1z2 = 0)
    # a=1, d=0, f=0 (x1x2 = 0) -> degenerate?
    
    polarities = [
        (1, 0, 0, 1, 0, 4), # x1x2 + y1y2 - z1z2
        (1, 0, 0, 1, 0, 1), # x1x2 + y1y2 + z1z2
        (1, 0, 0, 0, 0, 1), # x1x2 + z1z2
        (0, 1, 0, 1, 0, 0), # x1y2 + x2y1 + y1y2
        (1, 0, 1, 0, 0, 0), # x1x2 + x1z2 + x2z1
    ]
    
    for pol in polarities:
        if time.time() - start_time > DEADLINE:
            break
            
        adj, edges = build_polarity_graph(points, q, *pol)
        
        # Check if we have exactly 90 edges. If not, we might need to adjust.
        # The polarity graph of PG(2,5) is known to have 90 edges for the standard one.
        # Other polarities might have different edge counts.
        # We need exactly 90 edges in the base to add 1.
        # If edge count is not 90, we can try to add/remove edges to reach 90, 
        # but that breaks the "polarity" structure.
        # For now, only consider polarities that yield 90 edges.
        
        if len(edges) != 90:
            continue
            
        edge_set = set(edges)
        
        # Find best edge to add
        best_pair, cost = min_cost_edge(adj, edges, edge_set)
        
        if best_pair is None:
            continue
            
        # Add the edge
        u, v = best_pair
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append(best_pair)
        edge_set.add(best_pair)
        
        # Count C4s
        c4_count = count_c4s(adj, edges)
        
        if c4_count < best_total_c4:
            best_total_c4 = c4_count
            best_edges_final = list(edges)
            
    if best_edges_final is None:
        # Fallback to standard polarity
        adj, edges = build_polarity_graph(points, q, 1, 0, 0, 1, 0, 4)
        edge_set = set(edges)
        best_pair, cost = min_cost_edge(adj, edges, edge_set)
        if best_pair:
            u, v = best_pair
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append(best_pair)
            best_edges_final = list(edges)
        else:
            # Should not happen
            best_edges_final = edges
            
    return [[u, v] for u, v in best_edges_final]
# EVOLVE-BLOCK-END
