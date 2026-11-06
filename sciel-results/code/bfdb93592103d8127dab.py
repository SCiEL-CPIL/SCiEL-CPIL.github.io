<<<<<<< SEARCH
def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders using different heuristics."""
    orders = []
    
    # 1. Ascending degree (classic greedy for MIS)
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Ascending degree with random tie-breaking
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Dynamic min-degree
    orders.append(_dynamic_min_degree_order(A, n))
    
    # 4. Core number order
    orders.append(_core_order(A, deg, n))
    
    # 5. Spectral MWIS-inspired scores
    spectral_scores = _spectral_scores_mwis(A, n, rng, num_iters=50)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 6. Pure random
    orders.append(rng.permutation(n))
    
    # 7. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 8. Multiple random perturbations of degree order
    base_order = np.argsort(deg, kind='stable')
    for _ in range(2):
        perturbed = base_order.copy()
        for _ in range(n // 10):
            i, j = rng.integers(0, n, size=2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        orders.append(perturbed)
    
    # 9. 2-hop neighborhood size
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
    
    return orders
=======
def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders using different heuristics."""
    orders = []
    
    # 1. Ascending degree (classic greedy for MIS)
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Ascending degree with random tie-breaking
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Dynamic min-degree
    orders.append(_dynamic_min_degree_order(A, n))
    
    # 4. Core number order
    orders.append(_core_order(A, deg, n))
    
    # 5. Spectral MWIS-inspired scores
    spectral_scores = _spectral_scores_mwis(A, n, rng, num_iters=50)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 6. Pure random
    orders.append(rng.permutation(n))
    
    # 7. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 8. Multiple random perturbations of degree order
    base_order = np.argsort(deg, kind='stable')
    for _ in range(2):
        perturbed = base_order.copy()
        for _ in range(n // 10):
            i, j = rng.integers(0, n, size=2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        orders.append(perturbed)
    
    # 9. 2-hop neighborhood size
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
    
    # 10. Degree-weighted 2-hop: prefer vertices with low degree AND low 2-hop
    combined_2hop = deg + 0.3 * two_hop
    orders.append(np.argsort(combined_2hop, kind='stable'))
    
    # 11. Spectral with different alpha
    alpha2 = 0.7
    M2 = np.eye(n) - alpha2 * A.astype(float)
    x2 = rng.random(n)
    x2 = x2 / (np.linalg.norm(x2) + 1e-10)
    for _ in range(30):
        x2_new = M2 @ x2
        norm2 = np.linalg.norm(x2_new)
        if norm2 < 1e-10:
            break
        x2 = x2_new / norm2
    x2 = np.abs(x2)
    deg_norm2 = deg / (deg.max() + 1e-10)
    x2_mean = x2.mean()
    x2_std = x2.std() + 1e-10
    x2_norm = (x2 - x2_mean) / x2_std
    combined2 = deg_norm2 - 0.3 * x2_norm
    orders.append(np.argsort(combined2, kind='stable'))
    
    # 12. Max-neighbor-degree: prefer vertices whose neighbors have high degree
    max_nbr_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            max_nbr_deg[v] = deg[nbrs].max()
        else:
            max_nbr_deg[v] = 0
    orders.append(np.argsort(max_nbr_deg, kind='stable'))
    
    return orders
>>>>>>> REPLACE

