"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _nearest_neighbor_tour(dm, start=0):
    """
    Generate an initial tour using the nearest neighbor heuristic starting from a specific node.
    """
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    curr = start
    
    for _ in range(n - 1):
        # Get distances from current node to all unvisited nodes
        dists = dm[curr].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        curr = next_node
        
    return tour

def _cheapest_insertion_tour(dm, start=0):
    """
    Generate an initial tour using the cheapest insertion heuristic.
    1. Start with a tour of two nodes (start and its nearest neighbor).
    2. Repeatedly insert the unvisited node that has the minimum insertion cost.
    """
    n = dm.shape[0]
    if n <= 2:
        return list(range(n))
        
    # Find nearest neighbor to start
    dists = dm[start].copy()
    dists[start] = np.inf
    nn = np.argmin(dists)
    
    tour = [start, nn]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    visited[nn] = True
    
    while len(tour) < n:
        min_cost = float('inf')
        best_node = -1
        best_pos = -1
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Try inserting i into every possible position in the current tour
            # The tour is a list. Inserting between tour[j] and tour[(j+1)%len(tour)]
            # Cost = d(tour[j], i) + d(i, tour[(j+1)%len(tour)]) - d(tour[j], tour[(j+1)%len(tour)])
            for j in range(len(tour)):
                u = tour[j]
                v = tour[(j + 1) % len(tour)]
                cost = dm[u, i] + dm[i, v] - dm[u, v]
                if cost < min_cost:
                    min_cost = cost
                    best_node = i
                    best_pos = j + 1 # Insert after j, so at index j+1
            
        if best_node != -1:
            tour.insert(best_pos, best_node)
            visited[best_node] = True
        else:
            # Fallback if something goes wrong (shouldn't happen)
            for i in range(n):
                if not visited[i]:
                    tour.append(i)
                    visited[i] = True
                    break
                    
    return tour

def _two_opt(tour, dm):
    """
    Improve a tour using 2-opt local search.
    """
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                
                current_cost = dm[a, b] + dm[c, d]
                new_cost = dm[a, c] + dm[b, d]
                
                if new_cost < current_cost - 1e-10:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
    return tour

def _or_opt(tour, dm, segment_size=3):
    """
    Improve a tour using Or-Opt local search (move a segment of 1-3 nodes).
    """
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for size in range(1, segment_size + 1):
            for i in range(n):
                # Segment is tour[i] ... tour[i+size-1]
                # It is preceded by tour[i-1] and followed by tour[i+size]
                # We try to move it to a new position j
                
                # Extract segment
                seg = tour[i:i+size]
                if len(seg) < size:
                    continue
                    
                # Predecessor and successor in original position
                pred = tour[i-1]
                succ = tour[(i+size)%n]
                
                # Cost of removing segment
                # Original: pred->seg[0] ... seg[-1]->succ
                # New: pred->succ
                # Change in cost for removal: - (d(pred, seg[0]) + d(seg[-1], succ)) + d(pred, succ)
                
                # Try inserting segment between tour[j] and tour[j+1]
                # Skip inserting back in the same place or adjacent places that don't make sense
                for j in range(n):
                    # Avoid inserting in the same location
                    if j == i or j == (i-1)%n or j == (i-1+size)%n:
                         # Rough check to avoid trivial moves
                         # Specifically, if j is i, we are putting it back.
                         # If j is i-1, we are shifting by 0? No, inserting after i-1 means before i.
                         # Let's just compute and check improvement.
                         pass
                         
                    # If the segment is being moved to a position that overlaps with its original spot, 
                    # we need to be careful with indices if we modify the list.
                    # Easier: construct a new tour and compare cost? 
                    # For n=50, O(n^3) with small constants is fine.
                    # But let's try to be efficient.
                    
                    # Indices in the NEW tour (after removal)
                    # This is complex. Let's just do a simple O(n) check for improvement 
                    # by calculating the delta.
                    
                    # Let's stick to a simpler implementation for Or-Opt:
                    # Remove segment, then find best insertion point.
                    
                    # Actually, for n=50, a simple "remove and reinsert" check is fast enough.
                    # But we need to do it carefully.
                    
                    pass # Placeholder for complex logic, let's use a simpler robust approach below
                    
    # Simpler Or-Opt implementation
    tour = list(tour)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for size in range(1, 4):
            for i in range(n):
                # Segment starts at i
                if i + size > n: continue
                
                # Nodes in segment
                seg_nodes = tour[i:i+size]
                
                # Neighbors
                prev_node = tour[i-1]
                next_node = tour[(i+size)%n]
                
                # Cost of current connections
                cost_current = dm[prev_node, seg_nodes[0]] + dm[seg_nodes[-1], next_node]
                
                # Cost if we remove segment (connect prev to next)
                cost_removed = dm[prev_node, next_node]
                
                # Try inserting segment after node j
                # We must not insert it back in the same place
                best_delta = float('inf')
                best_j = -1
                
                for j in range(n):
                    # j is the node AFTER which we insert
                    # The segment will be between j and (j+1)%n
                    # But we have to account for the fact that the segment was removed.
                    # This is tricky with indices.
                    
                    # Let's just construct the new tour and calculate the delta for the affected edges.
                    # Affected edges:
                    # 1. prev_node -> seg[0] (removed)
                    # 2. seg[-1] -> next_node (removed)
                    # 3. j -> (j+1)%n (removed, unless j is part of the segment or neighbors)
                    # 4. prev_node -> next_node (added)
                    # 5. j -> seg[0] (added)
                    # 6. seg[-1] -> (j+1)%n (added)
                    
                    # We need to be careful if j is prev_node or next_node or inside the segment.
                    # If j == i-1 (prev_node), we are putting it back.
                    # If j == i+size-1 (last node of seg), we are putting it back.
                    
                    if j == i-1 or j == i+size-1:
                        continue
                        
                    # Also, if the insertion point overlaps with the segment, it's invalid.
                    # The segment occupies indices i ... i+size-1.
                    # If j is in [i, i+size-1], it's invalid.
                    if i <= j < i+size:
                        continue
                        
                    # Determine the node after j in the ORIGINAL tour
                    # But wait, if we remove the segment, the indices shift.
                    # Let's just build the new tour. It's O(n) per check, O(n^3) total. 
                    # For n=50, 125k ops * 3 sizes * few iterations is very fast.
                    
                    # Build new tour
                    new_tour = []
                    # Add nodes before i
                    new_tour.extend(tour[:i])
                    # Add nodes from i+size to end
                    new_tour.extend(tour[i+size:])
                    # Now insert seg after j
                    # Find index of j in new_tour? 
                    # No, j is an index in the ORIGINAL tour.
                    # This approach is getting messy.
                    
                    pass
    
    # Let's use a cleaner Or-Opt
    def _or_opt_clean(tour, dm):
        n = len(tour)
        tour = list(tour)
        improved = True
        while improved:
            improved = False
            for size in range(1, 4):
                for i in range(n):
                    # Segment: tour[i] to tour[i+size-1]
                    if i + size > n:
                        continue
                    
                    # Identify segment
                    seg = tour[i:i+size]
                    
                    # Identify neighbors
                    prev_idx = i - 1
                    next_idx = (i + size) % n
                    
                    prev_node = tour[prev_idx]
                    next_node = tour[next_idx]
                    
                    # Current cost contribution
                    cost_old = dm[prev_node, seg[0]] + dm[seg[-1], next_node]
                    
                    # Cost if removed
                    cost_remove = dm[prev_node, next_node]
                    
                    # Try inserting after every other node j
                    # We need to iterate through the tour excluding the segment
                    # Let's create a list of nodes to consider for insertion
                    # Nodes are 0..n-1. Segment is i..i+size-1.
                    # We want to insert between some node k and k+1.
                    
                    for j in range(n):
                        # j is the node BEFORE the insertion point
                        # Skip if j is in the segment
                        if i <= j < i+size:
                            continue
                        # Skip if j is the predecessor (would put it back)
                        if j == prev_idx:
                            continue
                        # Skip if j is the last node of the segment (would put it back)
                        # Wait, if we insert after the last node of the segment, that's the original position.
                        
                        # Determine the node AFTER j in the current tour
                        # If j is before i, next is j+1
                        # If j is after i+size-1, next is j+1
                        # But we have to handle wrap around and the gap.
                        
                        # Let's just calculate the delta directly.
                        # The edge (j, j+1) is broken.
                        # New edges: (j, seg[0]) and (seg[-1], j+1)
                        
                        # Find node after j
                        next_j_idx = (j + 1) % n
                        
                        # If next_j_idx is in the segment, this insertion is invalid/overlapping
                        if i <= next_j_idx < i+size:
                            continue
                            
                        # Also, if j is the node immediately before the segment (prev_idx), we skip.
                        # If j is the node immediately after the segment?
                        # If j == next_idx - 1 (which is i+size-1), that's in the segment.
                        
                        # What if j is next_idx?
                        # Then we insert between next_idx and next_idx+1.
                        
                        node_after_j = tour[next_j_idx]
                        
                        # Cost of edge to break
                        cost_break = dm[j, node_after_j]
                        
                        # Cost of new edges
                        cost_new = dm[j, seg[0]] + dm[seg[-1], node_after_j]
                        
                        # Total delta
                        # Remove: cost_old
                        # Add: cost_remove (connect prev and next)
                        # Break: cost_break
                        # Add: cost_new
                        delta = (cost_remove + cost_new) - (cost_old + cost_break)
                        
                        if delta < -1e-10:
                            # Perform the move
                            # 1. Remove segment
                            # 2. Insert segment after j
                            
                            # We need to be careful with indices.
                            # Let's reconstruct the tour.
                            # Original: [0..i-1, SEG, i+size..n-1]
                            # New: [0..j, SEG, j+1..i-1, i+size..n-1] (if j < i)
                            # Or similar.
                            
                            # Easier: 
                            # 1. Extract segment
                            # 2. Remove it from list
                            # 3. Find index of j in the NEW list
                            # 4. Insert segment after that index
                            
                            new_tour = list(tour)
                            # Remove segment
                            del new_tour[i:i+size]
                            
                            # Find index of j in new_tour
                            # j is a node ID, not an index. 
                            # Wait, j in the loop is an INDEX in the original tour.
                            # But after removal, indices shift.
                            # We should iterate over NODES, not indices, or handle shifts.
                            
                            # Let's iterate over nodes for j.
                            pass
                            
        return tour
        
    # Given the complexity of Or-Opt, and that 2-opt is very strong for n=50, 
    # and we are doing MULTIPLE STARTS, the diversity comes from the initial tours.
    # I will stick to 2-opt only to ensure robustness and speed, 
    # but run it on many diverse starts.
    
    return _two_opt(tour, dm)

def heuristics(distance_matrix):
    """
    Compute a heuristic matrix based on a consensus of multiple high-quality tours.
    """
    n = distance_matrix.shape[0]
    
    # 1. Generate multiple diverse tours
    tours = []
    
    # Start with deterministic NN from 0
    tours.append(_two_opt(_nearest_neighbor_tour(distance_matrix, 0), distance_matrix))
    
    # Start with Cheapest Insertion from 0
    tours.append(_two_opt(_cheapest_insertion_tour(distance_matrix, 0), distance_matrix))
    
    # Add randomized NN starts from different nodes
    random.seed(42) # Fixed seed for reproducibility in heuristic generation? 
                    # The task says "seeded per instance" for the ACO, but the heuristic 
                    # is computed once. It's better to be deterministic or use a fixed seed 
                    # so the heuristic is stable.
    
    for start_node in range(1, 10): # Try a few different start nodes
        # Randomize the order of nearest neighbors? 
        # Simple NN is deterministic. To get diversity, we can add random noise to distances.
        # Or just use different start nodes.
        
        # Let's try Cheapest Insertion with different starts too
        tours.append(_two_opt(_cheapest_insertion_tour(distance_matrix, start_node), distance_matrix))
        
        # And NN with different starts
        tours.append(_two_opt(_nearest_neighbor_tour(distance_matrix, start_node), distance_matrix))
        
    # 2. Aggregate edge frequencies
    # Count how often each edge appears in the consensus set of tours
    edge_counts = np.zeros((n, n), dtype=np.float64)
    
    for tour in tours:
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            edge_counts[u, v] += 1
            edge_counts[v, u] += 1
            
    # 3. Construct heuristic
    # Base: Inverse distance (or inverse square)
    safe_dm = distance_matrix + 1e-9
    base_heuristic = 1.0 / (safe_dm ** 2)
    
    # Boost edges that appear in the consensus
    # If an edge appears in k tours, multiply by (1 + k * boost_factor)
    # Or simply: heuristic = base_heuristic * (1 + edge_counts * boost)
    
    # Let's use a strong boost.
    # If an edge is in all 12 tours, it gets a huge boost.
    # If it's in 0 tours, it gets the base heuristic.
    
    # To avoid overwhelming the base heuristic for non-tour edges, 
    # we can normalize or use a specific formula.
    
    # Strategy: 
    # heuristic = base_heuristic * (1 + edge_counts * 10.0)
    
    heuristic = base_heuristic * (1.0 + edge_counts * 10.0)
    
    # Ensure positivity
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic

# EVOLVE-BLOCK-END
