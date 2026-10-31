"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_BUDGET = 140.0  # Safety margin: 150s limit, leave 10s for overhead/termination

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_BUDGET
    return _DEADLINE

def _path_cost(seq, dist_matrix):
    """Calculate the total cost of a path defined by the sequence of nodes."""
    if len(seq) < 2:
        return 0.0
    return np.sum(dist_matrix[seq[:-1], seq[1:]])

def _two_opt_improve(seq, dist_matrix, deadline):
    """
    Perform 2-Opt improvement on the sequence 'seq'.
    'seq' is a numpy array of node indices.
    Returns the improved sequence.
    """
    n = len(seq)
    if n < 4:
        return seq
    
    # Copy the sequence to avoid modifying the original
    best_seq = seq.copy()
    best_cost = _path_cost(best_seq, dist_matrix)
    
    improved = True
    passes = 0
    max_passes = 50  # Safety cap, though convergence is usually fast for N=50
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # Check deadline
        if time.time() > deadline - 2.0:
            break
            
        # We consider all pairs (i, j) with 0 <= i < j < n-1
        # Reversing best_seq[i+1:j+1]
        # Edges removed: (i, i+1) and (j, j+1)
        # Edges added: (i, j+1) and (i+1, j)
        
        # Vectorized calculation of deltas for all valid (i, j)
        # i ranges from 0 to n-3
        # j ranges from i+1 to n-2
        
        # To keep it simple and fast enough for N=50, we can use nested loops
        # but with NumPy for the specific cost lookups, or just pure Python loops
        # since N is small. 50^2 = 2500 checks per pass. Very fast.
        
        for i in range(n - 2):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n - 1):
                # Calculate delta
                # u = best_seq[i], v = best_seq[i+1]
                # x = best_seq[j], y = best_seq[j+1]
                
                d_ui = dist_matrix[best_seq[i], best_seq[i+1]]
                d_xj = dist_matrix[best_seq[j], best_seq[j+1]]
                d_ux = dist_matrix[best_seq[i], best_seq[j+1]]
                d_vy = dist_matrix[best_seq[i+1], best_seq[j]]
                
                delta = d_ux + d_vy - d_ui - d_xj
                
                if delta < -1e-10:
                    # Perform reversal
                    best_seq[i+1:j+1] = best_seq[i+1:j+1][::-1]
                    best_cost += delta
                    improved = True
                    # Restart inner loop or continue? 
                    # Standard 2-opt usually continues to find next improvement in the same pass
                    # or restarts. Continuing is faster.
                    
    return best_seq

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using 2-Opt refinement of the remaining path.
    
    Strategy:
    1. Check time budget. If low, fallback to Nearest Neighbor.
    2. Construct an initial path from current_node to destination_node 
       visiting all unvisited_nodes using Nearest Neighbor.
    3. Refine this path using 2-Opt local search.
    4. Return the first node in the refined path (the next city to visit).
    """
    deadline = _get_deadline()
    
    # Fast fallback if time is critical
    if time.time() > deadline - 5.0:
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        # Nearest Neighbor fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Exact solution for 2 nodes
        u0 = unvisited_nodes[0]
        u1 = unvisited_nodes[1]
        cost_0_1 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost_1_0 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        if cost_0_1 <= cost_1_0:
            return int(u0)
        else:
            return int(u1)

    if n_unvisited <= 10:
        # Brute force for small sets is possible but 10! is too big.
        # For <= 10, we can just run 2-Opt on the NN start, it will likely find optimal.
        # Or use Held-Karp? No, 2^10 * 10 is fine, but let's stick to 2-Opt for consistency.
        pass

    # 1. Initialize with Nearest Neighbor
    remaining = list(unvisited_nodes)
    order = []
    curr = current_node
    
    while remaining:
        # Find nearest in remaining
        dists = distance_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        order.append(next_node)
        remaining.pop(min_idx)
        curr = next_node
        
    # Sequence: current -> order[0] -> ... -> order[-1] -> destination
    # Note: We treat this as an open path from current to destination.
    # The 2-Opt will optimize the internal ordering.
    seq = np.array([current_node] + order + [destination_node], dtype=np.int64)
    
    # 2. Refine with 2-Opt
    # We only care about the internal nodes for the "next" choice, 
    # but optimizing the whole path gives the best context for the first step.
    optimized_seq = _two_opt_improve(seq, distance_matrix, deadline)
    
    # The next node is the second element in the sequence (index 1)
    # because index 0 is current_node.
    next_node_idx = 1
    if next_node_idx < len(optimized_seq):
        return int(optimized_seq[next_node_idx])
    else:
        # Fallback
        return int(unvisited_nodes[0])

# EVOLVE-BLOCK-END
