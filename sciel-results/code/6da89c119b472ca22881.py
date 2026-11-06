<<<<<<< SEARCH
def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders."""
    orders = []
    
    # 1. Static Ascending Degree
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Static Ascending Degree with Noise
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Dynamic Min-Degree
    orders.append(_dynamic_min_degree_order(A, n))
    
    # 4. Dynamic Max-Neighbor-Degree
    orders.append(_dynamic_max_neighbor_degree_order(A, n))
    
    # 5. Spectral (Low Centrality first)
    orders.append(_spectral_order(A, n, reverse=True))
    
    # 6. Spectral (High Centrality first)
    orders.append(_spectral_order(A, n, reverse=False))
    
    # 7. Random
    orders.append(rng.permutation(n))
    
    # 8. Core Number Order
    orders.append(_core_order(A, deg, n))
    
    # 9. Spectral power iteration based
    spectral_scores = _spectral_scores_power(A, n, rng, num_iters=20)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 10. 2-hop neighborhood size
    two_hop = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) == 0:
            two_hop[v] = 0
        else:
            two_hop_nbrs = set()
            for u in nbrs:
                two_hop_nbrs.update(np.where(A[u])[0])
            two_hop_nbrs.discard(v)
            two_hop_nbrs.difference_update(nbrs)
            two_hop[v] = len(two_hop_nbrs)
    orders.append(np.argsort(two_hop, kind='stable'))
    
    # 11. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 12. Perturbed degree order
    base_order = np.argsort(deg, kind='stable')
    perturbed = base_order.copy()
    for _ in range(n // 10):
        i, j = rng.integers(0, n, size=2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    orders.append(perturbed)
    
    # 13. Neighbor clustering score: prefer vertices whose neighbors are densely connected
    #     (picking such vertices blocks fewer potential future picks)
    neighbor_clustering = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) < 2:
            neighbor_clustering[v] = 0
        else:
            sub_A = A[np.ix_(nbrs, nbrs)]
            edges = sub_A.sum() / 2
            possible = len(nbrs) * (len(nbrs) - 1) / 2
            neighbor_clustering[v] = edges / possible if possible > 0 else 0
    
    # Combine: low degree + high neighbor clustering is good
    deg_norm = deg / (deg.max() + 1e-10)
    cluster_norm = neighbor_clustering / (neighbor_clustering.max() + 1e-10)
    combined_score = deg_norm - 0.5 * cluster_norm
    orders.append(np.argsort(combined_score, kind='stable'))
    
    # 14. Mean neighbor degree signal (from top-performing inspiration)
    mean_nbr_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            mean_nbr_deg[v] = deg[nbrs].mean()
        else:
            mean_nbr_deg[v] = 0
    
    mean_nbr_norm = mean_nbr_deg / (mean_nbr_deg.max() + 1e-10)
    deg_norm2 = deg / (deg.max() + 1e-10)
    combined2 = deg_norm2 + 0.3 * mean_nbr_norm
    orders.append(np.argsort(combined2, kind='stable'))
    
    return orders
=======
def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders."""
    orders = []
    
    # 1. Static Ascending Degree
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Static Ascending Degree with Noise
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Dynamic Min-Degree
    orders.append(_dynamic_min_degree_order(A, n))
    
    # 4. Dynamic Max-Neighbor-Degree
    orders.append(_dynamic_max_neighbor_degree_order(A, n))
    
    # 5. Spectral (Low Centrality first)
    orders.append(_spectral_order(A, n, reverse=True))
    
    # 6. Spectral (High Centrality first)
    orders.append(_spectral_order(A, n, reverse=False))
    
    # 7. Random
    orders.append(rng.permutation(n))
    
    # 8. Core Number Order
    orders.append(_core_order(A, deg, n))
    
    # 9. Spectral power iteration based
    spectral_scores = _spectral_scores_power(A, n, rng, num_iters=20)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 10. 2-hop neighborhood size
    two_hop = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) == 0:
            two_hop[v] = 0
        else:
            two_hop_nbrs = set()
            for u in nbrs:
                two_hop_nbrs.update(np.where(A[u])[0])
            two_hop_nbrs.discard(v)
            two_hop_nbrs.difference_update(nbrs)
            two_hop[v] = len(two_hop_nbrs)
    orders.append(np.argsort(two_hop, kind='stable'))
    
    # 11. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 12. Perturbed degree order
    base_order = np.argsort(deg, kind='stable')
    perturbed = base_order.copy()
    for _ in range(n // 10):
        i, j = rng.integers(0, n, size=2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    orders.append(perturbed)
    
    # 13. Neighbor clustering score: prefer vertices whose neighbors are densely connected
    neighbor_clustering = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) < 2:
            neighbor_clustering[v] = 0
        else:
            sub_A = A[np.ix_(nbrs, nbrs)]
            edges = sub_A.sum() / 2
            possible = len(nbrs) * (len(nbrs) - 1) / 2
            neighbor_clustering[v] = edges / possible if possible > 0 else 0
    
    deg_norm = deg / (deg.max() + 1e-10)
    cluster_norm = neighbor_clustering / (neighbor_clustering.max() + 1e-10)
    combined_score = deg_norm - 0.5 * cluster_norm
    orders.append(np.argsort(combined_score, kind='stable'))
    
    # 14. Mean neighbor degree signal
    mean_nbr_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            mean_nbr_deg[v] = deg[nbrs].mean()
        else:
            mean_nbr_deg[v] = 0
    
    mean_nbr_norm = mean_nbr_deg / (mean_nbr_deg.max() + 1e-10)
    deg_norm2 = deg / (deg.max() + 1e-10)
    combined2 = deg_norm2 + 0.3 * mean_nbr_norm
    orders.append(np.argsort(combined2, kind='stable'))
    
    # 15. Weighted degree order
    orders.append(_weighted_degree_order(A, n, deg))
    
    # 16. Combined: degree + inverse neighbor clustering
    inv_cluster = 1.0 - neighbor_clustering / (neighbor_clustering.max() + 1e-10)
    combined3 = deg_norm + 0.4 * inv_cluster
    orders.append(np.argsort(combined3, kind='stable'))
    
    return orders
>>>>>>> REPLACE

