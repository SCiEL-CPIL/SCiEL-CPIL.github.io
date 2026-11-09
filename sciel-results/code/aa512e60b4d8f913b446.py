import numpy as np
import time


def _nearest_neighbor_tour(dm, start=0):
    """Generate a tour using the nearest neighbor heuristic."""
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dm[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def _tour_length(tour, dm):
    """Calculate the length of a closed tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total


def _two_opt(tour, dm, max_iter=1000):
    """Improve a tour using 2-opt local search with delta calculation."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for reversing segment i+1..j
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
        # Early stop if no improvement in a full pass
        if not improved:
            break
    return tour, best_len


def _multi_start_two_opt(dm, n_starts=5, max_iter_per_start=200):
    """Run multiple 2-opt starts to find a good reference tour."""
    n = dm.shape[0]
    best_tour = None
    best_len = np.inf
    
    for s in range(n_starts):
        start_node = s % n
        tour = _nearest_neighbor_tour(dm, start=start_node)
        tour, length = _two_opt(tour, dm, max_iter=max_iter_per_start)
        if length < best_len:
            best_len = length
            best_tour = tour[:]
    
    return best_tour, best_len


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Uses a high-quality reference tour to boost heuristic values for those edges.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 330s limit
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # Base heuristic: inverse distance
    # Avoid division by zero
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Generate a high-quality reference tour
    # Use time budget to decide how much optimization to do
    # For n=50, 2-opt is fast. Let's do a few starts.
    ref_tour, ref_len = _multi_start_two_opt(dm, n_starts=5, max_iter_per_start=200)
    
    # Check time
    if time.time() > deadline:
        # Fallback to base heuristic if we ran out of time
        return np.maximum(base_heuristic, 1e-9)
    
    # Boost heuristic for edges in the reference tour
    # We want to make the ants prefer these edges.
    # Let's multiply the heuristic for these edges by a factor.
    # Or add a large constant? The task says "larger is more attractive".
    # Probability is proportional to pheromone * heuristic.
    # Pheromone starts at 1.
    # If we boost heuristic significantly, ants will likely follow the tour.
    # Let's try a multiplicative boost.
    
    heuristic = base_heuristic.copy()
    
    # Mark edges in the reference tour
    # The tour is a cycle: 0 -> ... -> n-1 -> 0
    for i in range(n):
        u = ref_tour[i]
        v = ref_tour[(i + 1) % n]
        # Boost both directions? The task says "in travel direction only" for pheromone,
        # but heuristic is usually symmetric in TSP.
        # However, the ant picks next node i->j.
        # If we boost heuristic[i,j], it makes j more attractive from i.
        # Since TSP is symmetric, dm[i,j] == dm[j,i], so base_heuristic is symmetric.
        # Let's boost the specific directed edges in the tour order.
        heuristic[u, v] *= 100.0
        
        # Also check time occasionally? No, this loop is small (n=50).
    
    # Ensure no values are not positive or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
