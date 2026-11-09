import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Try to find a valid graph for n=20, then 19, etc.
    # We use local search for each n.
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break

        # Try multiple initial random graphs for this n
        # Increase seeds for smaller n if 20 fails?
        # For now, try 100 seeds for 20, 50 for others.
        num_seeds = 100 if target_n >= 19 else 50
        
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break

            rng = random.Random(seed)
            n = target_n
            # Initialize adjacency matrix with random edges
            # Use 0.5 probability
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1

            # Local search loop
            # We try to eliminate violations
            # Limit iterations to avoid spending too much time on one seed
            max_iterations = 500
            for _ in range(max_iterations):
                if time.time() > deadline - 2:
                    break

                violation = _find_violation_numpy(adj, n)
                if violation is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}

                # Flip a random edge in the violating subset
                # This breaks the regularity of that specific subset
                # (though it might create new ones)
                # Ensure we pick distinct vertices
                if len(violation) < 2:
                    continue
                u, v = rng.sample(violation, 2)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]

    # Fallback: return a trivially valid small graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}

def _find_violation_numpy(adj, n):
    """
    Find any subset of size >= 6 that induces a regular graph.
    Returns the subset (as a tuple of indices) or None if valid.
    """
    if n < 6:
        return None

    # Check sizes from 6 to n
    # Smaller sizes are more likely to be regular?
    # Actually, size 6 is the most constrained.
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_induced_numpy(adj, subset):
                return subset
    return None

def _is_regular_induced_numpy(adj, subset):
    """
    Check if the induced subgraph on subset is regular using NumPy.
    """
    idx = np.array(subset)
    sub_adj = adj[np.ix_(idx, idx)]
    degs = sub_adj.sum(axis=1)
    return np.all(degs == degs[0])

