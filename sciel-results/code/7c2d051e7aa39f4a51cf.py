"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _build_bitmask_adj(n, edge_set):
    """Build adjacency as list of bitmasks and degree list."""
    adj_mask = [0] * n
    deg = [0] * n
    for u, v in edge_set:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    return adj_mask, deg

def _find_triangles_fast(n, adj_mask):
    """Find all triangles using bitmask operations."""
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_mask[u] >> v) & 1:
                continue
            # Common neighbors of u and v, with index > v
            common = adj_mask[u] & adj_mask[v] & (~((1 << (v + 1)) - 1))
            while common:
                w = (common & -common).bit_length() - 1
                triangles.append((u, v, w))
                common &= common - 1
    return triangles

def _score_graph(n, edge_set):
    """Compute the max degree sum over all triangles."""
    adj_mask, deg = _build_bitmask_adj(n, edge_set)
    triangles = _find_triangles_fast(n, adj_mask)
    if not triangles:
        return 0
    best = 0
    for u, v, w in triangles:
        s = deg[u] + deg[v] + deg[w]
        if s > best:
            best = s
    return best

def _soft_max_objective(n, edge_set, beta=2.0):
    """
    Smooth surrogate for max triangle degree sum using log-sum-exp.
    Also includes a mild penalty on variance to encourage uniformity.
    """
    adj_mask, deg = _build_bitmask_adj(n, edge_set)
    triangles = _find_triangles_fast(n, adj_mask)
    if not triangles:
        return 0.0
    
    sums = np.array([deg[u] + deg[v] + deg[w] for u, v, w in triangles], dtype=float)
    
    # Soft max via log-sum-exp
    m = sums.max()
    soft_max = m + np.log(np.exp(sums - m).sum()) / len(sums)
    
    # Mild variance penalty to encourage uniform triangle degree sums
    var_penalty = 0.05 * sums.var()
    
    return soft_max + var_penalty

def _get_worst_triangle_edges(n, edge_set):
    """Return edges that appear in the worst (highest degree-sum) triangles."""
    adj_mask, deg = _build_bitmask_adj(n, edge_set)
    triangles = _find_triangles_fast(n, adj_mask)
    if not triangles:
        return []
    
    # Find worst triangle
    worst_idx = 0
    worst_sum = -1
    for i, (u, v, w) in enumerate(triangles):
        s = deg[u] + deg[v] + deg[w]
        if s > worst_sum:
            worst_sum = s
            worst_idx = i
    
    u, v, w = triangles[worst_idx]
    return [(u, v), (u, w), (v, w)]

def construct():
    n = 16
    deadline = time.time() + 115  # Safety margin
    
    # Start with K_{8,8} + one intra-part edge
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((min(u, v), max(u, v)))
    edges.add((0, 1))
    
    best_edges = set(edges)
    best_score = _score_graph(n, best_edges)
    best_obj = _soft_max_objective(n, best_edges)
    
    # All possible pairs
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    random.seed(42)
    
    # Simulated annealing with soft-max objective and targeted moves
    T0 = 3.0
    T_end = 0.01
    cooling_rate = 0.99995
    
    T = T0
    current_edges = set(best_edges)
    current_obj = best_obj
    
    no_improve_count = 0
    max_no_improve = 8000
    
    while time.time() < deadline - 5:
        # Targeted move: 60% chance to remove edge from worst triangle,
        # 40% random
        if random.random() < 0.6:
            worst_edges = _get_worst_triangle_edges(n, current_edges)
            if worst_edges:
                removed = random.choice(worst_edges)
            else:
                removed = random.choice(tuple(current_edges))
        else:
            removed = random.choice(tuple(current_edges))
        
        non_edges = all_pairs - current_edges
        if not non_edges:
            continue
        
        # Targeted add: prefer edges that don't create new high-degree triangles
        # Use a simple heuristic: add random non-edge
        added = random.choice(tuple(non_edges))
        
        new_edges = current_edges - {removed} | {added}
        new_obj = _soft_max_objective(n, new_edges)
        
        delta_obj = new_obj - current_obj
        
        # Metropolis acceptance
        if delta_obj < 0 or (T > 0.001 and random.random() < math.exp(-delta_obj / T)):
            current_edges = new_edges
            current_obj = new_obj
            
            # Track best by actual score
            new_score = _score_graph(n, new_edges)
            if new_score < best_score:
                best_score = new_score
                best_edges = set(new_edges)
                best_obj = new_obj
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # If stuck for too long, restart temperature
        if no_improve_count > max_no_improve:
            T = T0
            no_improve_count = 0
    
    # Final local search polish: try all single-edge swaps from best_edges
    # Focus on removing edges from worst triangles
    polish_deadline = deadline - 2
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        worst_edges = _get_worst_triangle_edges(n, best_edges)
        for rem in worst_edges:
            if time.time() > polish_deadline:
                break
            non_edges = all_pairs - best_edges
            for add in non_edges:
                if time.time() > polish_deadline:
                    break
                new_edges = best_edges - {rem} | {add}
                new_score = _score_graph(n, new_edges)
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    improved = True
                    break
    
    # Convert to list format
    edge_list = [[u, v] for u, v in sorted(best_edges)]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
