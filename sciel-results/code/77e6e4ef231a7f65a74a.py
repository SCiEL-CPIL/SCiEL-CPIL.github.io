"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty with a repulsive force field around the current tour.
    Uses an additive penalty that decays exponentially with the "closeness" of the
    edge to the tour edges. Closeness is defined as the minimum over all tour edges
    of the sum of distances from the edge endpoints to the tour edge endpoints.
    """
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    D = edge_distance

    # Extract tour edges (i, i+1) cyclically
    a_arr = np.array([local_opt_tour[i] for i in range(n)], dtype=int)
    b_arr = np.array([local_opt_tour[(i + 1) % n] for i in range(n)], dtype=int)

    # Vectorized computation of min_closeness
    # A[i, t] = D[i, a_t]
    # B[i, t] = D[i, b_t]
    A = D[:, a_arr]  # shape (n, n)
    B = D[:, b_arr]  # shape (n, n)

    # Term1[i, j, t] = D[i, a_t] + D[j, b_t]
    # Term2[i, j, t] = D[i, b_t] + D[j, a_t]
    # We want min over t of min(Term1, Term2) for each (i, j)

    # To avoid creating a huge (n, n, n) array if n is large, we can loop,
    # but for n=100, (100, 100, 100) is 1e6 elements, which is small.
    # However, let's try to be memory efficient just in case.
    # Actually, 8MB is nothing. Let's use broadcasting for speed.

    Term1 = A[:, None, :] + B[None, :, :]  # shape (n, n, n)
    Term2 = B[:, None, :] + A[None, :, :]  # shape (n, n, n)

    MinTerm = np.minimum(Term1, Term2)
    min_closeness = np.min(MinTerm, axis=2)  # shape (n, n)

    # Parameters
    alpha = 1.0      # Base penalty magnitude
    beta = 10.0      # Decay rate
    gamma_used = 0.5 # Usage penalty coefficient

    # Calculate penalty
    # Exponential decay of closeness
    force_field = np.exp(-beta * min_closeness)

    # Modulate by usage
    # edge_n_used is a count matrix.
    usage_factor = 1.0 + gamma_used * edge_n_used

    penalty_matrix = alpha * force_field * usage_factor

    # Additive penalty
    result = D + penalty_matrix

    # Ensure symmetry and finite values
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)

    # Replace any non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=D.max() * 10, neginf=0.0)

    return result
# EVOLVE-BLOCK-END
