"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)

    # Mean degree of neighbors (a proxy for local density/clustering)
    sum_neighbor_degrees = adjacency.dot(degrees)
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]

    # Number of neighbors' neighbors that are also neighbors of the vertex (triangles involvement)
    # This is expensive for large n, so we approximate or skip if needed
    # For n=200, adjacency @ adjacency gives count of length-2 paths; diagonal is degree.
    # Triangles count = (adjacency @ adjacency - np.diag(degrees)) / 2 for each vertex? 
    # Actually, (A^2)[i,i] = degree(i). The number of triangles involving i is (A^2)[i,j] for j in neighbors... 
    # Let's compute triangle count per vertex: 
    # For each vertex i, triangles = sum over j in neighbors(i) of (number of common neighbors of i and j) / 2? No.
    # Simpler: triangles involving i = (A^2)[i, :][neighbors] sum? 
    # Let's just use a simpler heuristic that's fast.

    # Common heuristic: prefer vertices with low degree and low "clustering" tendency.
    # Another approach: use the "independent set size" proxy via greedy simulation on subsets? Too slow.

    # Let's try a combination:
    # 1. Low degree
    # 2. Low average neighbor degree (i.e., neighbors are not well-connected, so picking this vertex doesn't "waste" many potential picks)
    # Actually, if neighbors have high degree, they are "busy" and might not be picked anyway? 
    # But the greedy decoder picks in order. If a high-degree vertex is picked early, it blocks many.
    # So we want to pick vertices whose neighbors are less likely to be picked, i.e., neighbors with low priority.
    # This is circular. 

    # Alternative: Use the concept of "independent set density".
    # A vertex with low degree is good. A vertex whose neighbors are also low-degree is good? 
    # Because if neighbors are low-degree, they are less "valuable" to pick, so blocking them is less costly.

    # Let's define a score that combines:
    # - Low own degree
    # - Low sum of degrees of neighbors (so we don't block high-degree vertices that might be good picks)

    # Actually, the opposite: we want to block vertices that are "bad" picks.
    # A vertex with high degree is a bad pick (blocks many). So we want to block high-degree vertices.
    # So if our vertex's neighbors are high-degree, picking our vertex blocks many bad picks, which is good.
    # So we should prefer vertices with high average neighbor degree?

    # Let's test both. The seed uses -degree. 
    # Let's try: score = -degree + avg_neighbor_degree * weight
    # Or score = -degree / (1 + avg_neighbor_degree)

    # Let's try a simple combination:
    # score = -degrees + 0.5 * avg_neighbor_degree

    # Also, consider the "expansion" of the neighborhood.
    # Let's compute the number of unique neighbors of neighbors (excluding self and direct neighbors).
    # This is expensive. Let's skip for now.

    # Let's try multiple heuristics and pick the best one based on a quick simulation?
    # We have 145 seconds. We can simulate the greedy decoder on the full graph for a few heuristics and pick the best.

    def simulate_greedy(scores):
        """Simulate greedy decoder and return size of independent set."""
        n = len(scores)
        order = np.argsort(-scores)  # descending score
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as selected (blocked)
                neighbors = np.where(adjacency[v])[0]
                selected[neighbors] = True
        return count

    # Generate several candidate score vectors
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
        # Try a few more combinations
        extra_candidates = []
        for w in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]:
            extra_candidates.append(-degrees + w * avg_neighbor_degree)
        
        for cand in extra_candidates:
            if time.time() > deadline - 5:
                break
            sim_size = simulate_greedy(cand)
            if sim_size > best_score:
                best_score = sim_size
                best_candidate = cand

    return best_candidate
# EVOLVE-BLOCK-END
