import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    degrees = adjacency.sum(axis=1)
    
    # Precompute features
    sum_neighbor_degrees = adjacency.dot(degrees)
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]
    
    clustering = np.zeros(n)
    for i in range(n):
        if time.time() > deadline - 10:
            break
        d = int(degrees[i])
        if d > 1:
            neighbors = np.where(adjacency[i])[0]
            sub = adjacency[np.ix_(neighbors, neighbors)]
            edges = sub.sum() / 2.0
            possible = d * (d - 1) / 2.0
            clustering[i] = edges / possible

    # List of scoring functions
    def score_min_degree(adj, deg, avg_nd, clus):
        return -deg

    def score_current(adj, deg, avg_nd, clus):
        return avg_nd - deg + clus

    def score_avg_nd_minus_deg(adj, deg, avg_nd, clus):
        return avg_nd - deg

    def score_w_clus_10(adj, deg, avg_nd, clus):
        return avg_nd - deg + 10 * clus

    def score_w_clus_1(adj, deg, avg_nd, clus):
        return avg_nd - deg + 1 * clus

    def score_w_clus_5(adj, deg, avg_nd, clus):
        return avg_nd - deg + 5 * clus

    def score_w_clus_neg5(adj, deg, avg_nd, clus):
        return avg_nd - deg - 5 * clus

    def score_ratio(adj, deg, avg_nd, clus):
        safe_deg = deg + 1.0
        return avg_nd / safe_deg

    def score_inv_deg(adj, deg, avg_nd, clus):
        return 1.0 / (deg + 1.0)
        
    def score_avg_nd_only(adj, deg, avg_nd, clus):
        return avg_nd

    def score_mix_05(adj, deg, avg_nd, clus):
        return -0.5 * deg + 0.5 * avg_nd

    def score_mix_2_1(adj, deg, avg_nd, clus):
        return -2.0 * deg + 1.0 * avg_nd

    def score_mix_1_2(adj, deg, avg_nd, clus):
        return -1.0 * deg + 2.0 * avg_nd

    scorers = [
        score_min_degree,
        score_current,
        score_avg_nd_minus_deg,
        score_w_clus_10,
        score_w_clus_1,
        score_w_clus_5,
        score_w_clus_neg5,
        score_ratio,
        score_inv_deg,
        score_avg_nd_only,
        score_mix_05,
        score_mix_2_1,
        score_mix_1_2
    ]
    
    best_score_val = -1
    best_scores = None
    
    for scorer in scorers:
        if time.time() > deadline - 5:
            break
            
        scores = scorer(adjacency, degrees, avg_neighbor_degree, clustering)
        
        # Simulate greedy decoder
        order = np.argsort(-scores)
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if selected[v]:
                continue
            selected[v] = True
            count += 1
            neighbors = np.where(adjacency[v])[0]
            selected[neighbors] = True
            
        if count > best_score_val:
            best_score_val = count
            best_scores = scores

    if best_scores is None:
        return -degrees

    return best_scores
