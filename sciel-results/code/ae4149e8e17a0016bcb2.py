import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Generate points of P^2(5)
    # Points are (x, y, z) with x,y,z in 0..4, not all 0, normalized:
    # If x != 0, x=1. Else if y != 0, y=1. Else z=1.
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    # x is 1..4, invert x to make it 1
                    x_inv = pow(x, -1, 5)
                    points.append((x_inv * x % 5, x_inv * y % 5, x_inv * z % 5))
                elif y != 0:
                    y_inv = pow(y, -1, 5)
                    points.append((0, y_inv * y % 5, y_inv * z % 5))
                else:
                    z_inv = pow(z, -1, 5)
                    points.append((0, 0, z_inv * z % 5))
    
    # Ensure unique points (normalization should handle it)
    points = list(set(points))
    if len(points) != 31:
        # Fallback if normalization logic failed
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x==0 and y==0 and z==0: continue
                    # Just use raw tuples, dedupe later? No, normalization is standard.
                    # Let's re-verify normalization.
                    # (1, 2, 3) -> x=1, ok.
                    # (2, 4, 6=1) -> x=2, inv=3. (1, 2, 3). Same.
                    pass
        
    # 2. Define Polarity
    # Symmetric form: x1*x2 + y1*y2 + z1*z2
    def is_adjacent(p1, p2):
        return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5 == 0

    # 3. Build Graph
    edges = []
    adj = [[False]*31 for _ in range(31)]
    for i in range(31):
        for j in range(i+1, 31):
            if is_adjacent(points[i], points[j]):
                edges.append([i, j])
                adj[i][j] = True
                adj[j][i] = True
                
    current_edges = len(edges)
    
    # 4. Adjust Edge Count to 91
    # If current_edges > 91, remove edges.
    # If current_edges < 91, add edges.
    
    # Helper to count C4s
    def count_c4s(adj):
        # Count 4-cycles: a-b-c-d-a
        # Sum over pairs (a,c) of common neighbors b,d
        # C4 count = 1/4 * sum_{a,c} (C(common_neighbors(a,c), 2))
        # Or simply iterate a < b < c < d and check? No, too slow.
        # Use matrix multiplication or common neighbor counting.
        n = 31
        # adj is 0/1 matrix
        # Common neighbors of i and j is sum_k adj[i][k]*adj[j][k]
        # Let M be adjacency matrix.
        # (M^2)[i][j] = number of length-2 paths from i to j.
        # For i != j, (M^2)[i][j] is number of common neighbors.
        # A 4-cycle a-b-c-d involves pairs (a,c) and (b,d).
        # Number of C4s = 1/4 * sum_{i != j} C((M^2)[i][j], 2)
        # We can compute M^2 using numpy or manual loop.
        # Manual loop is fast enough for 31x31.
        m2 = [[0]*31 for _ in range(31)]
        for i in range(31):
            for k in range(31):
                if adj[i][k]:
                    for j in range(31):
                        if adj[k][j]:
                            m2[i][j] += 1
                            
        c4_count = 0
        for i in range(31):
            for j in range(i+1, 31):
                cn = m2[i][j]
                if cn >= 2:
                    c4_count += cn * (cn - 1) // 2
        return c4_count // 4

    # Optimization: If we need to remove edges, remove those in most C4s.
    # If we need to add edges, add those in fewest C4s (or 0).
    
    if current_edges > 91:
        # Remove excess
        excess = current_edges - 91
        # Calculate C4 contribution of each edge
        # This is expensive. Just remove random? No, we want min C4.
        # Greedy: Remove edge that is in most C4s.
        # Since graph is small, we can just try removing random edges and check?
        # Better: The polarity graph is very regular.
        # Let's just remove edges that are "redundant" for C4s.
        # Actually, if the base graph has 0 C4s, we can remove any.
        # If it has C4s, we should break them.
        
        # Let's check C4 count of base graph
        base_c4 = count_c4s(adj)
        
        if base_c4 == 0:
            # Just remove any edges
            for _ in range(excess):
                if time.time() > deadline: break
                # Remove last edge
                e = edges.pop()
                i, j = e
                adj[i][j] = False
                adj[j][i] = False
        else:
            # Greedy removal
            for _ in range(excess):
                if time.time() > deadline: break
                best_e = None
                best_c4_increase = -1
                # Try removing each edge
                # To speed up, we can just pick an edge at random or one with high degree
                # Given small size, we can even brute force the best removal
                for e in edges:
                    i, j = e
                    # Temporarily remove
                    adj[i][j] = False
                    adj[j][i] = False
                    new_c4 = count_c4s(adj)
                    # Restore
                    adj[i][j] = True
                    adj[j][i] = True
                    
                    # We want to minimize final C4. So we want removal that reduces C4 most (or keeps low).
                    # Actually, we just want the resulting graph to have min C4.
                    # So we pick e that minimizes new_c4.
                    if best_e is None or new_c4 < best_c4_increase:
                        best_c4_increase = new_c4
                        best_e = e
                        
                if best_e:
                    i, j = best_e
                    adj[i][j] = False
                    adj[j][i] = False
                    edges.remove(best_e)
                    
    elif current_edges < 91:
        # Add edges
        deficit = 91 - current_edges
        # Find non-edges
        non_edges = []
        for i in range(31):
            for j in range(i+1, 31):
                if not adj[i][j]:
                    non_edges.append([i, j])
        
        base_c4 = count_c4s(adj)
        
        for _ in range(deficit):
            if time.time() > deadline: break
            best_e = None
            best_c4 = float('inf')
            for e in non_edges:
                i, j = e
                adj[i][j] = True
                adj[j][i] = True
                new_c4 = count_c4s(adj)
                adj[i][j] = False
                adj[j][i] = False
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_e = e
                    
            if best_e:
                i, j = best_e
                adj[i][j] = True
                adj[j][i] = True
                non_edges.remove(best_e)
                edges.append(best_e)
                
    return edges

# EVOLVE-BLOCK-END
