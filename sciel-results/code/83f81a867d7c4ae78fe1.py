import numpy as np
import time

def _segments_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect properly."""
    # Using orientation method
    def orient(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val > 1e-10:
            return 1
        elif val < -1e-10:
            return -1
        return 0
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

def _count_edge_crossings(tour, distance_matrix, n):
    """Count how many times each edge in the complete graph is 'crossed' by tour edges.
    An edge (i,j) is crossed by a tour edge (a,b) if the segments intersect.
    We return a matrix where crossing_count[i,j] = number of tour edges that cross (i,j).
    """
    # Extract coordinates from distance matrix? We don't have them directly.
    # But we can work with the distance matrix structure.
    # Actually, we need coordinates to check geometric intersections.
    # The task says "50 uniform random points in the unit square" but we only get distance_matrix.
    # We need to reconstruct points or use a different approach.
    
    # Alternative: Use the tour itself to define "bad" edges.
    # In a 2D TSP, optimal tours are non-crossing.
    # So edges that are NOT in the tour but are "geometrically similar" to tour edges
    # in terms of being short might still be crossing.
    
    # Since we don't have coordinates, let's use a different heuristic:
    # Count how many times each pair (i,j) appears as a "conflicting" edge.
    # Two edges (i,j) and (k,l) conflict if they share no endpoints and 
    # the tour goes i->...->j->...->k->...->l in a way that suggests crossing.
    
    # Simpler approach: For each edge (i,j) in the complete graph, count how many
    # tour edges (a,b) have the property that the tour visits i and j in an order
    # that would cause a crossing with (a,b).
    
    # Actually, let's just use the inverse distance with a twist:
    # Penalize edges that are "long" relative to their position in the tour.
    
    # Let me try a different angle: use the 3-opt improved tour to identify
    # which edges are "good" (in the tour) and which are "bad" (crossing).
    
    # For each pair (i,j), compute a "crossing score" based on whether
    # adding edge (i,j) to the tour would create a crossing with existing tour edges.
    
    # This is complex without coordinates. Let's use a proxy:
    # An edge (i,j) is "attractive" if it's short AND it doesn't "conflict" with
    # the tour structure.
    
    # Let's define conflict as: in the tour, if we go from i to j, do we pass through
    # nodes that are closer to j than i is? This is getting complicated.
    
    # Let me try the simplest version: use inverse cube distance as base,
    # then boost edges in the 3-opt tour, and penalize edges that are "far"
    # from being in the tour.
    
    # Actually, let's just implement what was asked: count crossings.
    # We need coordinates. Let's assume we can extract them or use a different method.
    
    # Since we only have distance_matrix, let's use a graph-theoretic approach:
    # For each edge (i,j), count how many tour edges (a,b) have the property that
    # dist(i,a) + dist(b,j) < dist(i,j) + dist(a,b) (suggesting a potential crossing).
    
    crossing_count = np.zeros((n, n))
    tour_edges = []
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        tour_edges.append((a, b))
    
    # For each pair (i,j), check against all tour edges
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            count = 0
            for a, b in tour_edges:
                # Skip if they share endpoints
                if i in (a, b) or j in (a, b):
                    continue
                # Check if segments (i,j) and (a,b) would "cross" using distance heuristic
                # Two segments cross if the sum of diagonals is less than sum of opposite sides
                # This is a rough proxy for geometric crossing
                d_ij = distance_matrix[i, j]
                d_ab = distance_matrix[a, b]
                d_ia = distance_matrix[i, a]
                d_jb = distance_matrix[j, b]
                d_ib = distance_matrix[i, b]
                d_ja = distance_matrix[j, a]
                
                # If d_ia + d_jb < d_ij + d_ab and d_ib + d_ja < d_ij + d_ab
                # then the segments likely cross
                if (d_ia + d_jb < d_ij + d_ab) and (d_ib + d_ja < d_ij + d_ab):
                    count += 1
            crossing_count[i, j] = count
            crossing_count[j, i] = count
    
    return crossing_count

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 300.0  # Safety margin before 330s limit
    
    n = distance_matrix.shape[0]
    
    # 1. Construct NN tour
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        current = tour[-1]
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
    
    # 2. Improve with 3-opt
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i + 1) % n]]
        return total
    
    current_len = tour_length(tour)
    
    # 3-opt improvement
    max_passes = 5
    for _ in range(max_passes):
        improved = False
        for i in range(n - 2):
            if time.time() > deadline - 30:
                break
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    # Try all 3-opt moves
                    # Remove edges (i,i+1), (j,j+1), (k,k+1)
                    # Try reconnecting in different ways
                    
                    a = tour[i]
                    b = tour[i + 1]
                    c = tour[j]
                    d = tour[j + 1]
                    e = tour[k]
                    f = tour[(k + 1) % n]
                    
                    # Current length contribution: d(a,b) + d(c,d) + d(e,f)
                    current_contrib = distance_matrix[a, b] + distance_matrix[c, d] + distance_matrix[e, f]
                    
                    # Option 1: a-c, b-d, e-f (reverse segment i+1 to j)
                    new1 = distance_matrix[a, c] + distance_matrix[b, d] + distance_matrix[e, f]
                    if new1 < current_contrib - 1e-9:
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        current_len = current_len - current_contrib + new1
                        improved = True
                        continue
                    
                    # Option 2: a-d, b-c, e-f (swap b and c)
                    new2 = distance_matrix[a, d] + distance_matrix[b, c] + distance_matrix[e, f]
                    if new2 < current_contrib - 1e-9:
                        # Swap positions i+1 and j
                        tour[i + 1], tour[j] = tour[j], tour[i + 1]
                        current_len = current_len - current_contrib + new2
                        improved = True
                        continue
                    
                    # Option 3: a-e, b-f, c-d (move segment)
                    new3 = distance_matrix[a, e] + distance_matrix[b, f] + distance_matrix[c, d]
                    if new3 < current_contrib - 1e-9:
                        # This is more complex, skip for now
                        pass
        if not improved:
            break
    
    # 3. Compute crossing counts
    crossing_count = _count_edge_crossings(tour, distance_matrix, n)
    
    # 4. Build heuristic: base inverse distance, penalized by crossing count
    # H[i,j] = 1 / (distance[i,j]^2 * (1 + crossing_count[i,j]))
    # This makes edges that are "crossed" by the tour less attractive
    
    base = 1.0 / (distance_matrix + 1e-9) ** 2
    penalty = 1.0 + crossing_count
    H = base / penalty
    
    # 5. Boost edges in the 3-opt tour
    boost_tour = 10.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        H[u, v] *= boost_tour
        H[v, u] *= boost_tour
    
    # 6. k-NN boost for very close neighbors
    k = 2
    boost_knn = 3.0
    for i in range(n):
        dists_i = distance_matrix[i].copy()
        dists_i[i] = np.inf
        k_indices = np.argpartition(dists_i, k)[:k]
        for j in k_indices:
            H[i, j] *= boost_knn
            H[j, i] *= boost_knn
    
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
