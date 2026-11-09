import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.linalg import eigh
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 145.0

# Global state for the current instance
_STATE = {
    "coords": None,
    "dm_id": None,
    "path": []
}

def _recover_coordinates(distance_matrix):
    """
    Recover 2D coordinates from a Euclidean distance matrix using Classical MDS.
    """
    key = id(distance_matrix)
    if key in _STATE and _STATE["dm_id"] == key and _STATE["coords"] is not None:
        return _STATE["coords"]
    
    n = distance_matrix.shape[0]
    try:
        D2 = distance_matrix ** 2
        ones = np.ones((n, n))
        J = np.eye(n) - ones / n
        B = -0.5 * (J @ D2 @ J)
        
        vals, vecs = eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        coords = vecs[:, :2] * np.sqrt(np.maximum(vals[:2], 0))
        coords -= coords.mean(axis=0)
        
        _STATE["coords"] = coords
        _STATE["dm_id"] = key
        _STATE["path"] = []
        return coords
    except Exception:
        np.random.seed(42)
        coords = np.random.rand(n, 2)
        _STATE["coords"] = coords
        _STATE["dm_id"] = key
        _STATE["path"] = []
        return coords

def _segments_intersect(p1, p2, p3, p4):
    """
    Check if segment (p1, p2) intersects with segment (p3, p4).
    p1, p2, p3, p4 are 2D points (np arrays).
    """
    # Use orientation method
    def orient(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    # Collinear cases
    if o1 == 0 and np.dot(p3 - p1, p2 - p1) <= 0: return True
    if o2 == 0 and np.dot(p4 - p1, p2 - p1) <= 0: return True
    if o3 == 0 and np.dot(p1 - p3, p4 - p3) <= 0: return True
    if o4 == 0 and np.dot(p2 - p3, p4 - p3) <= 0: return True

    return False

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Ensure state is initialized for this distance matrix
    coords = _recover_coordinates(distance_matrix)
    
    # If this is the first call for this instance, initialize path
    if len(_STATE["path"]) == 0:
        _STATE["path"].append(current_node)
    
    # If current_node is not the last in path (should not happen if sequential),
    # we might need to reset or handle error. Assuming sequential.
    if _STATE["path"][-1] != current_node:
        # This might happen if the evaluator restarts or calls out of order.
        # Reset path to be safe, assuming current_node is the new start.
        _STATE["path"] = [current_node]

    # Small case: exact lookahead
    if n_unvisited <= 3:
        if n_unvisited == 2:
            u, v = unvisited_nodes
            cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
            cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
            best = int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
            _STATE["path"].append(best)
            return best
        else:
            u, v, w = unvisited_nodes
            perms = [(u, v, w), (u, w, v), (v, u, w), (v, w, u), (w, u, v), (w, v, u)]
            best_cost = float('inf')
            best_first = u
            for p in perms:
                c = (distance_matrix[current_node, p[0]] + 
                     distance_matrix[p[0], p[1]] + 
                     distance_matrix[p[1], p[2]] + 
                     distance_matrix[p[2], destination_node])
                if c < best_cost:
                    best_cost = c
                    best_first = p[0]
            _STATE["path"].append(best_first)
            return int(best_first)

    # General case: Greedy with Non-Crossing Check and Lookahead
    best_node = -1
    best_score = float('inf')
    
    current_coords = coords[current_node]
    
    # Pre-compute coordinates for unvisited
    unvisited_coords = coords[unvisited_nodes]
    
    # Existing edges in the path (excluding the last node which is 'current')
    # Path is [n0, n1, ..., nk-1, nk] where nk is current_node.
    # Edges are (n0,n1), (n1,n2), ..., (nk-1, nk).
    # We check if new edge (nk, v) crosses any of these.
    path_nodes = _STATE["path"]
    path_len = len(path_nodes)
    
    # If path length is 1, no edges to cross.
    # If path length > 1, check edges (path[i], path[i+1]) for i in 0..path_len-2
    
    for i, v in enumerate(unvisited_nodes):
        d_current_v = distance_matrix[current_node, v]
        
        # Lookahead: cost to next nearest from v
        # Exclude v from unvisited for lookahead
        if n_unvisited > 1:
            # Find min dist from v to other unvisited
            # We can do this quickly
            other_unvisited = np.delete(unvisited_nodes, i)
            min_next = np.min(distance_matrix[v, other_unvisited])
        else:
            min_next = distance_matrix[v, destination_node]
            
        base_score = d_current_v + 0.5 * min_next
        
        # Non-crossing check
        penalty = 0
        if path_len > 1:
            v_coords = coords[v]
            # Check against all existing edges
            # Optimization: only check if bounding boxes overlap? 
            # For 50 nodes, O(N) intersection checks is fine.
            for j in range(path_len - 1):
                p1 = coords[path_nodes[j]]
                p2 = coords[path_nodes[j+1]]
                # Check if (current_coords, v_coords) intersects (p1, p2)
                if _segments_intersect(current_coords, v_coords, p1, p2):
                    penalty = d_current_v * 10.0 + min_next * 5.0
                    break
        
        total_score = base_score + penalty
        
        if total_score < best_score:
            best_score = total_score
            best_node = v

    _STATE["path"].append(best_node)
    return int(best_node)
