"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _bitmask_score(adj_bits, n, edge_count):
    """
    Fast score using bitmask adjacency.
    adj_bits[i] is an int whose bit j is set iff edge (i,j) exists.
    Returns (max_triangle_deg_sum, triangle_list) where triangle_list is list of (u,v,w).
    """
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj_bits[i]).count('1')
    
    max_sum = 0
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_bits[u] >> v) & 1:
                continue
            # Common neighbors of u and v that are > v
            common = adj_bits[u] & adj_bits[v]
            # Only consider w > v
            mask = 0
            for w in range(v + 1, n):
                if (common >> w) & 1:
                    s = deg[u] + deg[v] + deg[w]
                    triangles.append((u, v, w))
                    if s > max_sum:
                        max_sum = s
    return max_sum, triangles

def _soft_max_from_triangles(triangles, deg, beta=2.0):
    """Compute soft-max objective from precomputed triangles and degrees."""
    if not triangles:
        return 0.0
    sums = [deg[u] + deg[v] + deg[w] for u, v, w in triangles]
    max_s = max(sums)
    if max_s == 0:
        return 0.0
    log_terms = [math.exp(beta * (s - max_s)) for s in sums]
    log_sum = sum(log_terms)
    if log_sum <= 0:
        return max_s
    soft_max = max_s + (1.0 / beta) * math.log(log_sum)
    mean_s = sum(sums) / len(sums)
    variance = sum((s - mean_s) ** 2 for s in sums) / len(sums)
    return soft_max + 0.05 * variance

def _spectral_init(n, num_edges, seed=42):
    """
    Use spectral method: build a random symmetric matrix, take top eigenvector,
    use it to rank edges by |x_u * x_v|, pick top num_edges.
    This tends to produce balanced graphs.
    """
    rng = np.random.RandomState(seed)
    # Random symmetric perturbation
    A = rng.randn(n, n)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0)
    
    # Add a small identity to break degeneracy
    A += 0.01 * np.eye(n)
    
    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # The top eigenvector gives a 2-coloring-like structure
    x = eigvecs[:, -1]  # top eigenvector
    
    # Score each potential edge by |x_u * x_v|
    edge_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            score = abs(x[i] * x[j])
            edge_scores.append((score, i, j))
    
    # Sort by score descending and take top num_edges
    edge_scores.sort(key=lambda t: -t[0])
    selected = [(i, j) for _, i, j in edge_scores[:num_edges]]
    return selected

def _local_search_polish(n, edges, deadline, max_iter=100000):
    """High-precision local search with targeted moves."""
    best_edges = set(edges)
    adj_bits = [0] * n
    for u, v in best_edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    best_score, _ = _bitmask_score(adj_bits, n, len(best_edges))
    
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    non_edges = list(all_pairs - best_edges)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        if not non_edges:
            break
        
        # Targeted: try removing edges from worst triangles, adding edges elsewhere
        deg = [bin(adj_bits[i]).count('1') for i in range(n)]
        _, triangles = _bitmask_score(adj_bits, n, len(best_edges))
        
        if triangles:
            # Find worst triangle
            worst = max(triangles, key=lambda t: deg[t[0]] + deg[t[1]] + deg[t[2]])
            worst_edges = [(worst[0], worst[1]), (worst[0], worst[2]), (worst[1], worst[2])]
            # Try removing an edge from the worst triangle
            if random.random() < 0.6:
                rem = random.choice(worst_edges)
                rem = (min(rem[0], rem[1]), max(rem[0], rem[1]))
                if rem not in best_edges:
                    rem = random.choice(list(best_edges))
            else:
                rem = random.choice(list(best_edges))
        else:
            rem = random.choice(list(best_edges))
        
        add = random.choice(non_edges)
        
        new_edges = best_edges - {rem} | {add}
        
        new_adj = [b for b in adj_bits]
        ru, rv = rem
        new_adj[ru] &= ~(1 << rv)
        new_adj[rv] &= ~(1 << ru)
        au, av = add
        new_adj[au] |= (1 << av)
        new_adj[av] |= (1 << au)
        
        new_score, _ = _bitmask_score(new_adj, n, len(new_edges))
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
            adj_bits = new_adj
            non_edges = list(all_pairs - best_edges)
    
    return best_edges, best_score

def construct():
    n = 16
    num_edges = 65
    start_time = time.time()
    deadline = start_time + 115
    
    random.seed(42)
    
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    # Spectral initialization
    spectral_edges = _spectral_init(n, num_edges, seed=42)
    spectral_set = set(spectral_edges)
    
    # Also try K_{8,8} + 1 as baseline
    k88_edges = set()
    for u in range(8):
        for v in range(8, 16):
            k88_edges.add((u, v))
    k88_edges.add((0, 1))
    
    # Evaluate both initializations
    def eval_init(edge_set):
        adj_b = [0] * n
        for u, v in edge_set:
            adj_b[u] |= (1 << v)
            adj_b[v] |= (1 << u)
        sc, _ = _bitmask_score(adj_b, n, num_edges)
        return sc
    
    sc_spectral = eval_init(spectral_set)
    sc_k88 = eval_init(k88_edges)
    
    if sc_spectral <= sc_k88:
        best_edges = set(spectral_set)
        best_score = sc_spectral
    else:
        best_edges = set(k88_edges)
        best_score = sc_k88
    
    # Build adj bits for current
    current_edges = set(best_edges)
    adj_bits = [0] * n
    for u, v in current_edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    current_score, _ = _bitmask_score(adj_bits, n, num_edges)
    
    # Simulated annealing
    T0 = 10.0
    T_end = 0.01
    cooling_rate = 0.99995
    T = T0
    beta = 1.0
    beta_max = 5.0
    no_improve_count = 0
    max_no_improve = 3000
    
    while time.time() < deadline - 15:
        # Generate move
        deg = [bin(adj_bits[i]).count('1') for i in range(n)]
        cur_score, triangles = _bitmask_score(adj_bits, n, num_edges)
        
        if triangles:
            worst = max(triangles, key=lambda t: deg[t[0]] + deg[t[1]] + deg[t[2]])
            worst_edges = [(worst[0], worst[1]), (worst[0], worst[2]), (worst[1], worst[2])]
            if random.random() < 0.5:
                rem = random.choice(worst_edges)
                rem = (min(rem[0], rem[1]), max(rem[0], rem[1]))
                if rem not in current_edges:
                    rem = random.choice(list(current_edges))
            else:
                rem = random.choice(list(current_edges))
        else:
            rem = random.choice(list(current_edges))
        
        non_edges_list = [p for p in all_pairs if p not in current_edges]
        if not non_edges_list:
            continue
        add = random.choice(non_edges_list)
        
        # Apply swap
        new_adj = [b for b in adj_bits]
        ru, rv = rem
        new_adj[ru] &= ~(1 << rv)
        new_adj[rv] &= ~(1 << ru)
        au, av = add
        new_adj[au] |= (1 << av)
        new_adj[av] |= (1 << au)
        
        new_score, new_tris = _bitmask_score(new_adj, n, num_edges)
        new_deg = [bin(new_adj[i]).count('1') for i in range(n)]
        new_obj = _soft_max_from_triangles(new_tris, new_deg, beta=beta)
        
        # Also compute current obj for comparison
        cur_obj = _soft_max_from_triangles(triangles, deg, beta=beta)
        
        delta_obj = new_obj - cur_obj
        
        if delta_obj < 0 or (T > 0 and random.random() < math.exp(-delta_obj / T)):
            current_edges = set(current_edges) - {rem} | {add}
            adj_bits = new_adj
            cur_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        if beta < beta_max:
            beta = min(beta_max, beta * 1.0001)
        
        if no_improve_count > max_no_improve:
            T = T0
            beta = 1.0
            no_improve_count = 0
    
    # Final polish
    polish_deadline = deadline - 2
    best_edges, best_score = _local_search_polish(n, best_edges, polish_deadline, max_iter=200000)
    
    edge_list = [[u, v] for u, v in sorted(best_edges)]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
