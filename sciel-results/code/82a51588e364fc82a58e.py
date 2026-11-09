import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s hard kill

def _tour_cost(tour, dm, start, end):
    """Calculate the total cost of a tour starting at 'start', going through 'tour', and ending at 'end'."""
    if len(tour) == 0:
        return dm[start, end]
    # d(start, tour[0]) + sum(d(tour[i], tour[i+1])) + d(tour[-1], end)
    cost = dm[start, tour[0]] + dm[tour[-1], end]
    if len(tour) > 1:
        cost += np.sum(dm[tour[:-1], tour[1:]])
    return cost

def _two_opt_improve(tour, dm, start, end, max_passes=20):
    """
    Vectorized 2-opt improvement.
    tour: list of node indices
    dm: distance matrix
    start: start node
    end: end node
    Returns improved tour.
    """
    n = len(tour)
    if n < 3:
        return tour
    
    current_tour = list(tour)
    
    for _ in range(max_passes):
        improved = False
        
        # Precompute edge costs for current tour
        # Edges: (start, t0), (t0, t1), ..., (t_{n-1}, end)
        # Indices in the "extended" tour [start] + tour + [end]
        
        # We look for i < j in the tour such that reversing tour[i..j] improves cost.
        # Old edges: (prev_i, t_i) and (t_j, next_j)
        # New edges: (prev_i, t_j) and (t_i, next_j)
        
        # Let's vectorize the delta calculation.
        # i ranges from 0 to n-2 (index in tour)
        # j ranges from i+1 to n-1 (index in tour)
        
        # prev_i: start if i==0 else tour[i-1]
        # next_j: end if j==n-1 else tour[j+1]
        
        # We can construct arrays of prev_nodes and next_nodes.
        prev_nodes = [start] + current_tour[:-1]
        next_nodes = current_tour[1:] + [end]
        
        # We need to evaluate delta for all pairs (i, j) with i < j.
        # This is an upper triangular matrix of deltas.
        
        # Let's use a loop over i, but vectorize over j. This is O(n^2) but with low constant factor.
        # Or fully vectorize using broadcasting?
        # For n=50, n^2 = 2500. Broadcasting 50x50 arrays is very fast.
        
        # Let's try full broadcasting.
        # i_idx = np.arange(n)
        # j_idx = np.arange(n)
        # I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        
        # Mask for i < j
        # mask = I < J
        
        # prev_I = prev_nodes[I]
        # t_I = current_tour[I]
        # t_J = current_tour[J]
        # next_J = next_nodes[J]
        
        # old_cost = dm[prev_I, t_I] + dm[t_J, next_J]
        # new_cost = dm[prev_I, t_J] + dm[t_I, next_J]
        # delta = new_cost - old_cost
        
        # We want the minimum delta < 0.
        
        # However, performing the reversal is not vectorizable easily.
        # So we find the best move, apply it, and repeat.
        
        # To avoid O(n^2) Python loop overhead for checking all pairs, we can just do the standard O(n^2) loop
        # but it's already implemented in the old code. The bottleneck might be the Python overhead of the loop itself.
        # Vectorizing the *search* for the best move is the key.
        
        # Let's implement the vectorized search for the best move.
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        prev_I = np.array(prev_nodes)[i_idx]
        t_I = np.array(current_tour)[i_idx]
        t_J = np.array(current_tour)[j_idx]
        next_J = np.array(next_nodes)[j_idx]
        
        old_cost = dm[prev_I, t_I] + dm[t_J, next_J]
        new_cost = dm[prev_I, t_J] + dm[t_I, next_J]
        delta = new_cost - old_cost
        
        min_idx = np.argmin(delta)
        min_delta = delta[min_idx]
        
        if min_delta < -1e-9:
            i = i_idx[min_idx]
            j = j_idx[min_idx]
            current_tour[i:j+1] = current_tour[i:j+1][::-1]
            improved = True
            
        if not improved:
            break
            
    return current_tour

def _or_opt_improve(tour, dm, start, end, max_passes=5):
    """
    Improve a tour using Or-opt (moving 1, 2, or 3 nodes).
    """
    n = len(tour)
    if n < 3:
        return tour
    
    current_tour = list(tour)
    
    for _ in range(max_passes):
        improved = False
        for seg_len in [1, 2, 3]:
            if n < seg_len + 2:
                continue
            for i in range(n - seg_len):
                # Extract segment
                seg = current_tour[i:i+seg_len]
                
                # Nodes around segment
                # Before: start if i==0 else current_tour[i-1]
                # After: end if i+seg_len==n else current_tour[i+seg_len]
                
                before_node = start if i == 0 else current_tour[i - 1]
                after_node = end if i + seg_len == n else current_tour[i + seg_len]
                
                # Cost of segment in place
                # Edges: (before, seg[0]) and (seg[-1], after)
                # Internal edges of seg: sum(dm[seg[k], seg[k+1]])
                internal_cost = 0
                if seg_len > 1:
                    internal_cost = np.sum(dm[seg[:-1], seg[1:]])
                
                old_local_cost = dm[before_node, seg[0]] + dm[seg[-1], after_node] + internal_cost
                
                # Try inserting segment at position k
                # k is index in the remaining list (excluding the segment)
                # The remaining list is current_tour[:i] + current_tour[i+seg_len:]
                # Let's just iterate over possible insertion points in the original tour indices
                # To avoid complexity, we iterate k from 0 to n-seg_len, skipping i..i+seg_len-1
                
                for k in range(n - seg_len + 1):
                    # Skip if k is within the segment being moved
                    if k >= i and k < i + seg_len:
                        continue
                    if k == i:
                        continue # Same position, no change (though logic below handles it)
                    
                    # Determine insertion point
                    # We are moving seg to be at index k in the new list?
                    # Actually, it's easier to think of inserting into the list `current_tour` at index k, 
                    # but we must account for the shift if k > i.
                    
                    # Let's construct the new tour and calculate cost delta? 
                    # That's O(n) per move. Or-opt has O(n^3) moves. 
                    # For n=50, 50^3 = 125k moves. 125k * O(1) delta is fine.
                    
                    # Let's calculate delta properly.
                    # If k < i:
                    #   Insert before k.
                    #   New edges: (node_before_k, seg[0]), (seg[-1], node_k)
                    #   Removed edges: (before_node, seg[0]), (seg[-1], after_node)
                    #   Also, the edge (node_before_k, node_k) is broken and replaced?
                    #   No, we are moving a segment.
                    #   Original: ... A, [Seg], B ...
                    #   Move Seg to before K: ... K, [Seg], K_next ...
                    #   Edges changed:
                    #   Old: (A, S0), (S_last, B), (K, K_next)
                    #   New: (A, B), (K, S0), (S_last, K_next)
                    
                    if k < i:
                        node_k = current_tour[k]
                        node_k_next = current_tour[k+1] if k+1 < n else end
                        node_a = before_node
                        node_b = after_node
                        
                        # Check if k is adjacent to i (i.e., k == i-1)
                        # If k == i-1, then node_k is before_node, and node_k_next is seg[0]
                        # This case is trivial (no move). We skip k==i-1? 
                        # Actually if k == i-1, we are moving it to the same place.
                        if k == i - 1:
                            continue
                        
                        old_cost = dm[node_a, seg[0]] + dm[seg[-1], node_b] + dm[node_k, node_k_next]
                        new_cost = dm[node_a, node_b] + dm[node_k, seg[0]] + dm[seg[-1], node_k_next]
                        
                    else: # k > i
                        # Node k is after the segment.
                        # We need to adjust index because removing segment shifts indices.
                        # In the original list, index k is at k.
                        # If we remove segment at i, the element at k becomes at k - seg_len.
                        # We are inserting at position k in the *new* list?
                        # Standard Or-opt: insert into the list *without* the segment.
                        # Let's just use the original indices for clarity.
                        
                        # If k > i + seg_len - 1:
                        node_k = current_tour[k]
                        # The node after k in the original list
                        node_k_next = current_tour[k+1] if k+1 < n else end
                        
                        # If we insert after node_k (at position k+1 in original indices)
                        # Wait, standard definition: insert segment such that it follows node at index k.
                        
                        # Old edges: (A, S0), (S_last, B), (K, K_next)
                        # New edges: (A, B), (K, S0), (S_last, K_next)
                        
                        # Note: If k == i + seg_len, then node_k is after_node.
                        # If k == i + seg_len, then B is node_k.
                        # Old: (A, S0), (S_last, K), (K, K_next)
                        # New: (A, K), (K, S0), (S_last, K_next)
                        
                        node_a = before_node
                        node_b = after_node
                        
                        if k == i + seg_len:
                            # Moving to immediately after itself? No, k is the index of the node AFTER the segment.
                            # If we insert after node_k, and node_k is the one after the segment, 
                            # we are moving it to the same place?
                            # Actually, if k == i + seg_len, we are inserting after B.
                            # This is valid.
                            pass
                            
                        old_cost = dm[node_a, seg[0]] + dm[seg[-1], node_b] + dm[node_k, node_k_next]
                        new_cost = dm[node_a, node_b] + dm[node_k, seg[0]] + dm[seg[-1], node_k_next]

                    if new_cost < old_cost - 1e-9:
                        # Perform move
                        # Remove segment
                        new_tour = current_tour[:i] + current_tour[i+seg_len:]
                        # Insert at k
                        # If k < i, insert at k
                        # If k > i, insert at k - seg_len (because indices shifted)
                        if k < i:
                            insert_idx = k
                        else:
                            insert_idx = k - seg_len
                        
                        new_tour[insert_idx:insert_idx] = seg
                        current_tour = new_tour
                        improved = True
                        break # Restart inner loop or break to next k?
                        # Breaking inner loop is safer to avoid index errors
                        # We break the k loop, but continue checking other i?
                        # Usually, we restart the whole pass or continue carefully.
                        # Let's break the k loop and let the i loop continue, but current_tour changed.
                        # This is dangerous. Let's just break all loops and restart pass.
                        # For simplicity and safety in this rewrite, we'll just do 2-opt which is safer.
                        # Or-opt is complex to implement correctly with indices.
                        # Given time constraints, I will stick to a very robust 2-opt and maybe 3-opt.
                        # Actually, let's just use 2-opt with more passes.
                        break
                if improved:
                    break
            if improved:
                break
                
    return current_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    dm = distance_matrix
    c = current_node
    dest = destination_node
    cands = unvisited_nodes
    
    # Exact for 2 nodes
    if n_unvisited == 2:
        u, v = cands
        cost_u = dm[c, u] + dm[u, v] + dm[v, dest]
        cost_v = dm[c, v] + dm[v, u] + dm[u, dest]
        return int(cands[0] if cost_u <= cost_v else cands[1])
    
    # Exact for 3 nodes
    if n_unvisited == 3:
        best_cost = np.inf
        best_node = cands[0]
        for first in range(3):
            rest = [i for i in range(3) if i != first]
            for second in rest:
                third = rest[0] if rest[1] == second else rest[1]
                c_val = (dm[c, cands[first]] +
                         dm[cands[first], cands[second]] +
                         dm[cands[second], cands[third]] +
                         dm[cands[third], dest])
                if c_val < best_cost:
                    best_cost = c_val
                    best_node = cands[first]
        return int(best_node)
    
    # For n up to 50, try all candidates with strong local search
    if n_unvisited <= 50 and time.time() - _START_TIME < _DEADLINE - 10:
        best_cost = np.inf
        best_node = cands[0]
        
        for first_idx in range(n_unvisited):
            first_node = cands[first_idx]
            rest = [cands[i] for i in range(n_unvisited) if i != first_idx]
            
            # Build NN tour
            sub_tour = []
            cur2 = first_node
            rem2 = list(rest)
            while rem2:
                d_to = dm[cur2, rem2]
                idx2 = int(np.argmin(d_to))
                nxt2 = rem2[idx2]
                sub_tour.append(nxt2)
                rem2 = rem2[:idx2] + rem2[idx2+1:]
                cur2 = nxt2
            
            # Improve with 2-opt
            # Use more passes since it's vectorized
            sub_tour = _two_opt_improve(sub_tour, dm, first_node, dest, max_passes=30)
            
            total = _tour_cost([first_node] + sub_tour, dm, c, dest)
            
            if total < best_cost:
                best_cost = total
                best_node = first_node
                
            # Time check
            if time.time() - _START_TIME > _DEADLINE - 5:
                break
        
        return int(best_node)
    
    # Fallback for very large n or time pressure
    # Just pick nearest neighbor
    return int(cands[np.argmin(dm[c, cands])])
