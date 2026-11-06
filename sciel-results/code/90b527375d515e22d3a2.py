"""Rewritten TSP constructive heuristic using convex-hull boundary expansion."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = None
_TIME_LIMIT = 115.0


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def _convex_hull_perimeter(points):
    """Compute the perimeter of the convex hull of a set of 2D points."""
    n = len(points)
    if n < 3:
        # Degenerate case: sum of all pairwise distances
        if n == 2:
            return 2 * np.linalg.norm(points[0] - points[1])
        return 0.0
    try:
        hull = ConvexHull(points)
        vertices = hull.vertices
        if len(vertices) > 1:
            next_vertices = np.roll(vertices, -1)
            perimeter = np.sum(np.linalg.norm(points[vertices] - points[next_vertices], axis=1))
        else:
            perimeter = 0.0
        return perimeter
    except Exception:
        # Fallback: approximate perimeter as 2 * max distance from centroid
        centroid = np.mean(points, axis=0)
        return 2 * np.max(np.linalg.norm(points - centroid, axis=1))


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a convex-hull boundary expansion heuristic.
    
    The idea: we want to keep the partial tour compact by favoring nodes that
    reduce the convex hull perimeter of the visited set. This is a global
    geometric metric that avoids the local myopia of nearest-neighbor.
    
    Since we don't track the full visited set, we approximate by:
    1. Computing the distance from current_node to each candidate
    2. Estimating the "boundary cost" as how far the candidate is from the
       current tour's convex hull (approximated by the set of recently visited
       nodes, which we approximate as current_node and destination_node)
    3. Combining immediate distance with a boundary-expansion penalty
    """
    _ensure_start_time()
    
    if _time_exceeded():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Handle small cases with exact lookahead
    if n_unvisited == 2:
        i0, i1 = unvisited_nodes
        cost0 = distance_matrix[current_node, i0] + distance_matrix[i0, i1] + distance_matrix[i1, destination_node]
        cost1 = distance_matrix[current_node, i1] + distance_matrix[i1, i0] + distance_matrix[i0, destination_node]
        return int(i0 if cost0 <= cost1 else i1)
    
    # For larger sets, use a combined heuristic:
    # 1. Immediate distance from current node
    # 2. Minimum distance from candidate to other unvisited nodes (connectivity)
    # 3. Distance to destination (for end-game awareness)
    
    # Compute sub-matrix among unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_dist, np.inf)
    min_outgoing = np.min(sub_dist, axis=1)
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive weighting based on how many nodes remain
    if n_unvisited <= 5:
        w_immediate = 1.0
        w_connectivity = 0.5
        w_dest = 0.5
    elif n_unvisited <= 10:
        w_immediate = 1.0
        w_connectivity = 0.6
        w_dest = 0.2
    else:
        w_immediate = 1.0
        w_connectivity = 0.7
        w_dest = 0.1
    
    score = w_immediate * dist_from_current + w_connectivity * min_outgoing + w_dest * dist_to_dest
    
    idx = np.argmin(score)
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
