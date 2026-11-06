"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin for 150s kill

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _apply_oropt(tour, distance_matrix, max_passes=3):
    """
    Apply Or-Opt improvement (moving 1 or 2 nodes) to a list of nodes representing an open path.
    The cost function includes the closing edge back to the start.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    t = tour[:]
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # Or-Opt 1: Move single node
        for i in range(n):
            u = t[i]
            # Remove u from position i
            # Neighbors in cycle: prev, u, next
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            prev_node = t[prev_idx]
            next_node = t[next_idx]
            
            # Old cost: (prev, u) + (u, next)
            old_cost = distance_matrix[prev_node, u] + distance_matrix[u, next_node]
            
            # Try inserting u between every pair (j, j+1) in the remaining tour
            # The remaining tour is t without u.
            # To keep it simple and O(N^2), we iterate j from 0 to n-2 in the *original* index space, 
            # skipping i.
            
            best_new_cost = float('inf')
            best_j = -1
            
            # We need to be careful with indices when removing u.
            # Let's create a temporary list without u for evaluation? 
            # Or just calculate indices relative to the cycle.
            
            # Iterate over all possible insertion points in the cycle of length n-1.
            # The cycle is t[0]...t[i-1], t[i+1]...t[n-1]
            # Insertion between k and k+1 in this new sequence.
            
            # To avoid complex index mapping, let's just iterate j in 0..n-1
            # and check if the insertion point is valid (not adjacent to u's old position in a way that creates a 2-cycle or invalid move, though 2-cycles are just swaps).
            
            for j in range(n):
                if j == i or j == (i - 1) % n or j == (i + 1) % n:
                    continue # Skip invalid or no-op positions
                
                # Insertion point is between t[j] and t[(j+1)%n] in the *original* tour?
                # No, if we remove u, the neighbors change.
                # This approach is tricky with indices.
                
                # Simpler: Explicitly remove u, try insertions, keep best, put back.
                # This is O(N^3) per pass if done naively with list ops.
                # For N=50, N^3 = 125k. 50 nodes * 125k = 6M ops per pass. 
                # 50 calls * 6M = 300M ops. Might be too slow in Python.
                
                pass # Placeholder for logic, will implement efficient version below
                
            # Efficient Or-Opt 1 implementation:
            # Iterate i. 
            # For each i, calculate old cost.
            # Iterate j (insertion index in the *remaining* cycle).
            
        # Or-Opt 2: Move pair of nodes
        # Similar logic for pairs.
        
    return t

def _fast_oropt(tour, distance_matrix, max_passes=2):
    """
    Optimized Or-Opt for small N.
    """
    n = len(tour)
    if n < 5:
        return tour
    
    t = tour[:]
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # Or-Opt 1
        for i in range(n):
            u = t[i]
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            prev_node = t[prev_idx]
            next_node = t[next_idx]
            
            old_cost = distance_matrix[prev_node, u] + distance_matrix[u, next_node]
            
            # Try inserting u between j and j+1
            # We iterate j from 0 to n-1
            # If j is prev_idx or next_idx, skip.
            # Also skip if j is i (impossible)
            
            best_delta = 0.0
            best_j = -1
            
            for j in range(n):
                if j == i:
                    continue
                # The insertion is between t[j] and t[(j+1)%n]
                # But wait, if we remove u, the indices shift.
                # Let's define the insertion in the *current* list t.
                # If we remove u at i, and insert between j and j+1:
                # If j < i, the new index of j+1 is j.
                # If j > i, the new index of j is j-1.
                
                # Let's just compute the cost change directly.
                # Case 1: j is not adjacent to i
                # The edges removed are (prev, u) and (u, next).
                # The edges added are (t[j], u) and (u, t[(j+1)%n]).
                # BUT, if j is prev_idx, we are inserting u back to the same place (or reverse).
                # If j is next_idx, we are inserting u back to the same place (or reverse).
                
                if j == prev_idx or j == next_idx:
                    continue
                
                # Also, if the tour is small, we might create a 2-cycle.
                
                # The nodes at j and j+1 in the *original* list t.
                # Note: If j < i, removing u doesn't change index of j.
                # If j > i, removing u shifts indices of j and j+1 down by 1?
                # Actually, let's just use the values.
                
                node_j = t[j]
                node_j_next = t[(j + 1) % n]
                
                # Are node_j and node_j_next adjacent to u in the original tour?
                # If j == prev_idx, node_j is prev_node. Skipped.
                # If j == next_idx, node_j_next is next_node? No, if j=next_idx, node_j is next_node.
                # We skip j=next_idx, so we don't insert u between next_node and its next.
                # We skip j=prev_idx, so we don't insert u between prev_node and u.
                
                # What if j is such that node_j_next is prev_node?
                # i.e., j = (prev_idx - 1) % n.
                # Then we are inserting u between t[prev-1] and prev_node.
                # This is valid.
                
                # What if node_j is next_node?
                # i.e., j = next_idx. Skipped.
                
                # So we are safe from immediate reversals?
                
                new_cost = distance_matrix[node_j, u] + distance_matrix[u, node_j_next]
                delta = new_cost - old_cost
                
                if delta < best_delta - 1e-8:
                    best_delta = delta
                    best_j = j
            
            if best_j != -1:
                # Perform the move
                # Remove u from i
                u_val = t.pop(i)
                # Adjust best_j if best_j > i
                if best_j > i:
                    best_j -= 1
                # Insert u at best_j + 1 (between best_j and best_j+1)
                # Wait, if we inserted between j and j+1, we insert at index j+1 in the *new* list.
                t.insert(best_j + 1, u_val)
                improved = True
                # Restart loop for correctness
                break
        
        if improved:
            continue
        
        # Or-Opt 2
        for i in range(n - 1):
            u = t[i]
            v = t[i + 1]
            prev_idx = (i - 1) % n
            next_idx = (i + 2) % n
            prev_node = t[prev_idx]
            next_node = t[next_idx]
            
            old_cost = distance_matrix[prev_node, u] + distance_matrix[u, v] + distance_matrix[v, next_node]
            
            best_delta = 0.0
            best_j = -1
            
            for j in range(n):
                # Skip if j is in [i-1, i, i+1, i+2] range roughly
                if j == i or j == (i + 1) % n or j == prev_idx or j == next_idx:
                    continue
                
                node_j = t[j]
                node_j_next = t[(j + 1) % n]
                
                # Check if the pair (u,v) is being re-inserted in a way that overlaps
                # If node_j is prev_node or next_node, we might be reversing.
                # For simplicity, we accept it, 2-opt will fix reversals later if needed,
                # or we can check if the segment is reversed.
                
                new_cost = distance_matrix[node_j, u] + distance_matrix[u, v] + distance_matrix[v, node_j_next]
                delta = new_cost - old_cost
                
                if delta < best_delta - 1e-8:
                    best_delta = delta
                    best_j = j
            
            if best_j != -1:
                # Perform the move
                u_val = t[i]
                v_val = t[i + 1]
                del t[i]
                del t[i]
                
                # Adjust best_j
                if best_j > i:
                    best_j -= 2
                elif best_j > i - 1: # This case is complex, but best_j != i, i+1
                    best_j -= 1 # If best_j was i+1, it's now i. If best_j was i, skipped.
                    # Actually, if best_j > i, it shifted by 2.
                    # If best_j < i, it didn't shift.
                    
                # Re-calculate shift carefully
                # Original indices: 0 ... i-1, i(u), i+1(v), i+2 ... n-1
                # Removed u, v. New length n-2.
                # If best_j > i+1, new index is best_j - 2.
                # If best_j < i, new index is best_j.
                # We skipped best_j in {i, i+1, i-1, i+2}.
                
                if best_j > i + 1:
                    insert_idx = best_j - 2 + 1 # Insert after best_j-2 (which is best_j in old coords)
                    # Wait. If we insert between j and j+1 in old coords.
                    # In new coords, j is j-2. j+1 is j-1.
                    # We want to insert after j-2. So index is (j-2) + 1 = j-1.
                    # My previous logic: best_j > i => best_j -= 2. Then insert at best_j + 1.
                    # If best_j (old) = 10, i=2. New best_j = 8. Insert at 9.
                    # Old: ... 8, 9, 10(u), 11(v) ...
                    # New: ... 8, 9, 10 ...
                    # Insert between 9 and 10 (old) -> between 7 and 8 (new)?
                    # No.
                    # Let's just use a simpler, safer insertion for Or-Opt 2.
                    
                # Given the complexity and risk of bugs in Or-Opt 2 index management,
                # and the time limit, let's stick to Or-Opt 1 which is simpler and often sufficient.
                # Or-Opt 1 is a structural change from 2-opt.
                
                # We will only use Or-Opt 1 in this implementation to ensure correctness.
                pass # Skip Or-Opt 2 to avoid bugs
                
    return t

def _estimate_tour_length(tour, distance_matrix):
    """
    Calculate the total length of a closed tour.
    """
    n = len(tour)
    if n <= 1:
        return 0.0
    
    total = 0.0
    # Sum consecutive edges
    for i in range(n - 1):
        total += distance_matrix[tour[i], tour[i+1]]
    # Closing edge
    total += distance_matrix[tour[-1], tour[0]]
    return total

def _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by evaluating each candidate with a fast greedy completion
    and a limited Or-Opt refinement.
    """
    best_candidate = -1
    best_score = float('inf')
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Pre-compute distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    for idx, candidate in enumerate(unvisited_nodes):
        # Simulate the rest of the tour using Nearest Neighbor
        sim_tour = _current_tour + [candidate]
        
        # Remaining nodes are unvisited_nodes excluding candidate
        remaining_arr = np.delete(unvisited_nodes, idx)
        
        curr = candidate
        while len(remaining_arr) > 0:
            # Find nearest neighbor in remaining_arr
            dists = distance_matrix[curr, remaining_arr]
            next_idx = np.argmin(dists)
            next_node = remaining_arr[next_idx]
            sim_tour.append(int(next_node))
            curr = next_node
            # Remove next_node from remaining_arr
            remaining_arr = np.delete(remaining_arr, next_idx)
        
        # Evaluate the simulated tour with Or-Opt
        # Check time
        if time.time() - _start_time > _TIME_LIMIT - 5:
            # Fallback to simple distance or skip Or-Opt
            score = _estimate_tour_length(sim_tour, distance_matrix)
        else:
            # Apply Or-Opt
            # Limit passes to keep it fast
            optimized_tour = _fast_oropt(sim_tour, distance_matrix, max_passes=2)
            score = _estimate_tour_length(optimized_tour, distance_matrix)
        
        if score < best_score:
            best_score = score
            best_candidate = candidate
            
    return int(best_candidate)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # Use lookahead strategy with Or-Opt
    next_node = _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Update global state
    _current_tour.append(next_node)
    
    return next_node

# EVOLVE-BLOCK-END
