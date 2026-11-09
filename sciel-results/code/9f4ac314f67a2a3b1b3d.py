# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin below the 120 s wall-clock limit


def _nearest_fallback(current_node, unvisited_nodes, distance_matrix):
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return destination_node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Deadline guard: if we are running low on wall-clock time, fall back to
    # a cheap nearest-neighbour choice so we always return in time.
    if time.time() - _START_TIME > _DEADLINE:
        return _nearest_fallback(current_node, unvisited_nodes, distance_matrix)

    # Work directly on the coordinate matrix: each row of distance_matrix is
    # a 2-D coordinate (Euclidean instances).
    coords = np.asarray(distance_matrix, dtype=np.float64)

    # Centroid of the unvisited set (the sweep pivot).
    pivot = coords[unvisited_nodes].mean(axis=0)

    # Vectors from the pivot to each unvisited node.
    vecs = coords[unvisited_nodes] - pivot

    # The reference direction is the vector from the pivot to the current node
    # (the node we are leaving). This anchors the sweep so the tour continues
    # "around" the current position rather than jumping back.
    ref = coords[current_node] - pivot
    ref_norm = np.linalg.norm(ref)

    if ref_norm < 1e-12:
        # Degenerate: current node is at the pivot; sweep from +x axis.
        ref = np.array([1.0, 0.0])
        ref_norm = 1.0
    ref = ref / ref_norm

    # Angle of each candidate relative to the reference direction, wrapped to
    # [0, 2*pi) so we always move forward around the pivot.
    cos_a = (vecs @ ref) / (np.linalg.norm(vecs, axis=1) + 1e-12)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    cross = vecs[:, 0] * ref[1] - vecs[:, 1] * ref[0]
    angles = np.arctan2(cross, cos_a)
    angles = np.mod(angles, 2.0 * np.pi)

    # Pick the candidate with the smallest positive angular advance (the next
    # node in the sweep). This yields a star-shaped, low-crossing tour.
    idx = int(np.argmin(angles))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
