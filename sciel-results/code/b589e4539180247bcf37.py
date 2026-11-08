"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit

    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)

    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]

    # Mean degree of neighbors
    sum_neighbor_degrees = adjacency.dot(degrees)
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]

    # Triangle count per vertex
    A_sq = adjacency @ adjacency
    triangle_count = np.zeros(n)
    for i in range(n):
        if degrees[i] > 0:
            triangle_count[i] = 0.5 * A_sq[i, neighbor_lists[i]].sum()

    # Compute "effective degree": sum of degrees of neighbors
    # This is already computed as sum_neighbor_degrees

    # Compute "neighborhood independence": how many pairs of neighbors are NOT connected
    # For vertex i with degree d, max pairs = d*(d-1)/2
    # Connected pairs = triangle_count[i] * 2 (since each triangle contributes 2 edges among neighbors of i)
    # Actually, triangle_count[i] = number of triangles involving i
    # Each triangle (i,j,k) means j and k are both neighbors of i and j-k is an edge
    # So number of edges among neighbors of i = triangle_count[i]
    # Number of non-edges among neighbors of i = d*(d-1)/2 - triangle_count[i]
    
    neighborhood_independence = np.zeros(n)
    for i in range(n):
        d = degrees[i]
        if d >= 2:
            neighborhood_independence[i] = d * (d - 1) / 2.0 - triangle_count[i]

    def simulate_greedy(scores):
        """Simulate greedy decoder and return size of independent set."""
        order = np.argsort(-scores)  # descending score
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as selected (blocked)
                selected[neighbor_lists[v]] = True
        return count

    # Generate candidate score vectors
    candidates = []

    # 1. Seed: -degree
    candidates.append(-degrees)

    # 2. -degree + avg_neighbor_degree
    candidates.append(-degrees + avg_neighbor_degree)

    # 3. -degree / (1 + avg_neighbor_degree)
    candidates.append(-degrees / (1 + avg_neighbor_degree))

    # 4. -degree * (1 + avg_neighbor_degree)
    candidates.append(-degrees * (1 + avg_neighbor_degree))

    # 5. Random perturbation of degree to break ties
    rng = np.random.RandomState(42)
    candidates.append(-degrees + rng.normal(0, 0.1, n))

    # 6. -degree + 2 * avg_neighbor_degree
    candidates.append(-degrees + 2 * avg_neighbor_degree)

    # 7. -degree * avg_neighbor_degree
    candidates.append(-degrees * avg_neighbor_degree)

    # 8. -degree + 0.5 * avg_neighbor_degree
    candidates.append(-degrees + 0.5 * avg_neighbor_degree)

    # 9. -degree + triangle_count
    candidates.append(-degrees + triangle_count)

    # 10. -degree + 0.5 * triangle_count
    candidates.append(-degrees + 0.5 * triangle_count)

    # 11. -degree - triangle_count (prefer low triangle count)
    candidates.append(-degrees - triangle_count)

    # 12. -degree / (1 + triangle_count)
    candidates.append(-degrees / (1 + triangle_count))

    # 13. -degree + avg_neighbor_degree + triangle_count
    candidates.append(-degrees + avg_neighbor_degree + triangle_count)

    # 14. -degree + 0.5 * avg_neighbor_degree + 0.5 * triangle_count
    candidates.append(-degrees + 0.5 * avg_neighbor_degree + 0.5 * triangle_count)

    # 15. -degree + 0.2 * avg_neighbor_degree + 0.8 * triangle_count
    candidates.append(-degrees + 0.2 * avg_neighbor_degree + 0.8 * triangle_count)

    # 16. -degree + neighborhood_independence
    candidates.append(-degrees + neighborhood_independence)

    # 17. -degree + 0.5 * neighborhood_independence
    candidates.append(-degrees + 0.5 * neighborhood_independence)

    # 18. -degree + avg_neighbor_degree + neighborhood_independence
    candidates.append(-degrees + avg_neighbor_degree + neighborhood_independence)

    # 19. -degree / (1 + neighborhood_independence)
    candidates.append(-degrees / (1 + neighborhood_independence))

    # 20. -degree + 0.3 * avg_neighbor_degree + 0.7 * neighborhood_independence
    candidates.append(-degrees + 0.3 * avg_neighbor_degree + 0.7 * neighborhood_independence)

    # Evaluate each candidate
    best_score = -np.inf
    best_candidate = candidates[0]

    for cand in candidates:
        if time.time() > deadline - 5:
            break
        sim_size = simulate_greedy(cand)
        if sim_size > best_score:
            best_score = sim_size
            best_candidate = cand

    # If we have time, try to optimize further with local search or more heuristics
    if time.time() < deadline - 10:
        # Try a grid search over weights for avg_neighbor_degree, triangle_count, and neighborhood_independence
        weights_an = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        weights_tc = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        weights_ni = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

        # Do a 3D grid search but limit to manageable size
        # Try a subset: fix one weight and vary the other two
        for w_an in weights_an:
            if time.time() > deadline - 5:
                break
            for w_tc in weights_tc:
                if time.time() > deadline - 5:
                    break
                cand = -degrees + w_an * avg_neighbor_degree + w_tc * triangle_count
                sim_size = simulate_greedy(cand)
                if sim_size > best_score:
                    best_score = sim_size
                    best_candidate = cand

        # Now try with neighborhood_independence
        for w_an in [0.0, 0.5, 1.0]:
            if time.time() > deadline - 5:
                break
            for w_ni in weights_ni:
                if time.time() > deadline - 5:
                    break
                cand = -degrees + w_an * avg_neighbor_degree + w_ni * neighborhood_independence
                sim_size = simulate_greedy(cand)
                if sim_size > best_score:
                    best_score = sim_size
                    best_candidate = cand

    return best_candidate
# EVOLVE-BLOCK-END
