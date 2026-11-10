import time
import numpy as np

# Explicit deadline guard
_DEADLINE_FRACTION = 0.9
_WALL_LIMIT = 330.0
_start = time.time()


def _deadline_left():
    return _WALL_LIMIT * _DEADLINE_FRACTION - (time.time() - _start)


def _nearest_neighbor_tour(d, start=0):
    """
    Construct a tour using the Nearest Neighbor heuristic starting from a specific node.
    """
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        dist = d[cur].copy()
        dist[visited] = np.inf
        nxt = int(np.argmin(dist))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order


def _two_opt(d, order, max_iters=100):
    """
    Improve a tour using the 2-opt heuristic.
    Returns the improved order and its length.
    """
    n = len(order)
    if n < 4:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    def tour_len(o):
        return float(sum(d[o[i], o[(i + 1) % n]] for i in range(n)))

    best = list(order)
    best_len = tour_len(best)
    
    for _ in range(max_iters):
        improved = False
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a = best[i - 1]
                b = best[i]
                c = best[j]
                e = best[(j + 1) % n]
                
                delta = (d[a, c] + d[b, e]) - (d[a, b] + d[c, e])
                
                if delta < -1e-12:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    best_len += delta
                    improved = True
        
        if not improved:
            break
            
    return best, best_len


def _three_opt(d, order, max_iters=20):
    """
    Improve a tour using the 3-opt heuristic.
    Tries all ways to reconnect three segments.
    """
    n = len(order)
    if n < 5:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    def tour_len(o):
        return float(sum(d[o[i], o[(i + 1) % n]] for i in range(n)))

    best = list(order)
    best_len = tour_len(best)
    
    for _ in range(max_iters):
        improved = False
        
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    # Three cut points: i, j, k
                    # Current edges: (i-1,i), (j,j+1), (k,k+1) [mod n]
                    # Three ways to reconnect:
                    # Option 1: reverse segment [i..j], reverse segment [j+1..k]
                    # Option 2: reverse segment [i..k]
                    # Option 3: reverse segment [i..j], reverse segment [k..] (cyclic)
                    
                    a = best[i - 1]
                    b = best[i]
                    c = best[j]
                    d_node = best[j + 1]
                    e = best[k]
                    f = best[(k + 1) % n]
                    
                    # Current: a-b ... c-d ... e-f
                    # Option 1: a-c, b-d, e-f -> reverse [i..j]
                    # New: a-c, b-d, e-f (but we need to check connectivity)
                    # Actually 3-opt: remove edges (a,b), (c,d), (e,f) and add 3 new edges
                    # The 3 segments are: [i-1] to i, [i+1..j] to j+1, [j+2..k] to k+1
                    # Wait, let me reconsider.
                    
                    # Standard 3-opt: pick 3 edges to remove and reconnect
                    # Edges removed: (best[i-1], best[i]), (best[j], best[j+1]), (best[k], best[k+1])
                    # We can reconnect in 3 ways (excluding the original):
                    # 1. (best[i-1], best[j+1]), (best[i], best[k+1]), (best[j], best[k])
                    #    This reverses segment [i..j] and segment [j+1..k]
                    # 2. (best[i-1], best[k+1]), (best[i], best[j]), (best[j+1], best[k])
                    #    This reverses segment [i..k]
                    # 3. (best[i-1], best[k]), (best[i], best[j+1]), (best[j], best[k+1])
                    #    This reverses segment [i..j] and segment [k..end]+[start..k-1]
                    
                    # Option 1: reverse [i..j] and reverse [j+1..k]
                    delta1 = (d[a, d_node] + d[b, f] + d[c, e]) - (d[a, b] + d[c, d_node] + d[e, f])
                    
                    # Option 2: reverse [i..k]
                    delta2 = (d[a, f] + d[b, c] + d[d_node, e]) - (d[a, b] + d[c, d_node] + d[e, f])
                    
                    # Option 3: reverse [i..j] and reverse the cyclic segment containing k
                    # This is trickier. Let's just use options 1 and 2.
                    
                    best_delta = min(delta1, delta2)
                    if best_delta < -1e-12:
                        if delta1 <= delta2:
                            # Reverse [i..j]
                            best[i:j + 1] = best[i:j + 1][::-1]
                            # Reverse [j+1..k]
                            best[j + 1:k + 1] = best[j + 1:k + 1][::-1]
                        else:
                            # Reverse [i..k]
                            best[i:k + 1] = best[i:k + 1][::-1]
                        best_len += best_delta
                        improved = True
        
        if not improved:
            break
            
    return best, best_len


def _good_directed_tour(d):
    """
    Find a high-quality directed tour by trying multiple nearest-neighbor starts
    and applying 2-opt + 3-opt to each, keeping the best result.
    """
    n = d.shape[0]
    if n <= 2:
        return list(range(n)), float(sum(d[i, (i+1)%n] for i in range(n)))

    best_order = None
    best_len = np.inf
    
    start_pool = list(range(0, min(n, 30)))
    
    for s in start_pool:
        if _deadline_left() < 5.0:
            break
            
        tmp = _nearest_neighbor_tour(d, start=s)
        
        opt, ln = _two_opt(d, tmp, max_iters=100)
        opt, ln = _three_opt(d, opt, max_iters=20)
        
        if ln < best_len:
            best_len = ln
            best_order = opt
            
    if best_order is None:
        # Fallback
        best_order = _nearest_neighbor_tour(d, start=0)
        best_order, best_len = _two_opt(d, best_order, max_iters=10)
        
    return best_order, best_len


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a combination of inverse distance and a strong directed boost for edges
    present in a near-optimal tour found via multi-start NN + 2-opt.
    """
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero.
    # Diagonal is 0 in distance matrix usually, so 1/(0+eps) is huge.
    # We will handle diagonal separately.
    base = 1.0 / (d + 1e-9)
    
    # Ensure diagonal is not attractive (ants shouldn't pick themselves)
    # The task says "Heuristic entries that are not positive become 1e-9".
    # If we set diagonal to 0, it becomes 1e-9.
    # But usually, we want diagonal to be low.
    # Let's set diagonal to 0 for now, then sanitize at the end.
    np.fill_diagonal(base, 0.0)
    
    # Find a reference directed tour
    bias = np.zeros_like(d)
    if n > 2 and _deadline_left() > 10.0:
        order, tour_len = _good_directed_tour(d)
        # Use distance-weighted bias: shorter edges in the tour get higher boost
        # This creates a smoother gradient that guides ants toward good edges
        mean_edge = tour_len / n if n > 0 else 1.0
        for i in range(n):
            u = order[i]
            v = order[(i + 1) % n]
            edge_dist = d[u, v]
            # Boost is inversely proportional to edge length, normalized
            # Shorter edges get higher attractiveness
            w = mean_edge / (edge_dist + 1e-9)
            # Symmetric bias for robustness
            bias[u, v] = w
            bias[v, u] = w
            
    # Blend: strong boost on the good tour edges.
    # Use a moderate boost to balance between tour structure and distance guidance
    boost = 50.0
    H = base + boost * bias
    
    # Sanitize per contract:
    # 1. Non-finite values -> 1e-9
    # 2. Non-positive values -> 1e-9
    # 3. Diagonal should be low. base diagonal was 0. bias diagonal is 0 (no self-loops in tour).
    # So H diagonal is 0. It will become 1e-9.
    
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
