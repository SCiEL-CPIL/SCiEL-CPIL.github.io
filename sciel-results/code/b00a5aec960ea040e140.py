"""Force-directed TSP constructive heuristic with 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_T0 = None
_DEADLINE = 140.0  # Safety margin before 150s kill
_SAFETY = 10.0

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def _two_opt_improve(tour, distance_matrix):
    """Apply 2-opt local search to improve the tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if _elapsed() > _DEADLINE - _SAFETY:
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta cost
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    # Reverse the segment between i+1 and j
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a force-directed model with 2-opt improvement.
    
    Strategy:
    1. Generate candidate next nodes using force-directed scoring
    2. For each candidate, simulate the remaining tour and apply 2-opt
    3. Return the candidate that leads to the best overall tour
    
    Force-directed scoring:
    - Attraction to current node (short edge)
    - Repulsion from previously visited nodes (spreading)
    - Attraction to destination (closing the tour)
    """
    unvisited_nodes = np.asarray(unvisited_nodes)
    n_unvisited = unvisited_nodes.size
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Hard deadline guard
    if _elapsed() > _DEADLINE - _SAFETY:
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
    
    try:
        N_total = distance_matrix.shape[0]
        progress = (N_total - n_unvisited) / max(N_total - 1, 1)
        progress = float(np.clip(progress, 0.0, 1.0))
        
        # Force-directed scoring
        d_cur = distance_matrix[current_node, unvisited_nodes]
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # Repulsion from previously visited nodes (approximated by average distance to all nodes)
        # This encourages spreading out
        avg_dist = np.mean(distance_matrix[unvisited_nodes, :], axis=1)
        
        # Score: minimize distance to current, maximize repulsion (spread), minimize distance to destination
        # We want to minimize: d_cur - alpha * avg_dist + beta * d_dest
        # Where alpha increases with progress (more spreading as we go), beta increases late
        alpha = 0.2 + 0.3 * progress
        beta = progress ** 2
        
        score = d_cur - alpha * avg_dist + beta * d_dest
        
        # Get top K candidates
        K = min(5, n_unvisited)
        top_indices = np.argsort(score)[:K]
        candidates = unvisited_nodes[top_indices]
        
        # For each candidate, estimate the quality of the resulting tour
        best_candidate = candidates[0]
        best_score = float('inf')
        
        for cand in candidates:
            if _elapsed() > _DEADLINE - _SAFETY:
                break
            
            # Simulate: current -> cand -> remaining unvisited (greedy) -> destination
            remaining = unvisited_nodes[unvisited_nodes != cand]
            
            # Build a quick tour estimate
            tour_est = [current_node, cand]
            curr = cand
            rem_list = list(remaining)
            
            while rem_list:
                if _elapsed() > _DEADLINE - _SAFETY:
                    break
                d_to_rem = distance_matrix[curr, rem_list]
                next_idx = np.argmin(d_to_rem)
                next_node = rem_list[next_idx]
                tour_est.append(next_node)
                curr = next_node
                rem_list.pop(next_idx)
            
            tour_est.append(destination_node)
            
            # Apply 2-opt to this estimated tour
            tour_np = np.array(tour_est)
            tour_improved = _two_opt_improve(tour_np.tolist(), distance_matrix)
            
            # Calculate final tour cost
            total_cost = 0.0
            for i in range(len(tour_improved)):
                total_cost += distance_matrix[tour_improved[i]][tour_improved[(i + 1) % len(tour_improved)]]
            
            if total_cost < best_score:
                best_score = total_cost
                best_candidate = cand
        
        return int(best_candidate)
        
    except Exception:
        # Fallback to simple NN
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
# EVOLVE-BLOCK-END
