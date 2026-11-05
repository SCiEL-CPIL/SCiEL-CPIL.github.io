import random
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_INDEX = {}
for i, edge in enumerate(P['edge_order']):
    u, v = edge
    if u > v:
        u, v = v, u
    EDGE_INDEX[(u, v)] = i + 1  # 1-based index

def get_edge_index(u, v):
    if u > v:
        u, v = v, u
    return EDGE_INDEX.get((u, v), 10**9)

def cycle_max_index(cycle):
    max_idx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        idx = get_edge_index(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def two_opt(cycle, i, j):
    # Reverses the segment between i and j (exclusive of endpoints in standard 2-opt logic, 
    # but here we define the move as removing edges (i-1, i) and (j, j+1) and adding (i-1, j) and (i, j+1))
    # Let's use a simpler representation: 
    # Current edges: (c[i-1], c[i]) and (c[j], c[j+1])
    # New edges: (c[i-1], c[j]) and (c[i], c[j+1])
    # This requires reversing the segment c[i..j]
    # To maintain a list permutation, we can just swap the order of the segment.
    # However, for efficiency, we can calculate the new max index without modifying the list first.
    pass

def optimize_cycle(cycle, time_limit=5.0):
    start_time = time.time()
    current_max = cycle_max_index(cycle)
    best_cycle = cycle[:]
    best_max = current_max
    
    # Precompute adjacency lists of "good" edges? 
    # Instead, we'll use a greedy 2-opt approach.
    # We look for the edge with the maximum index in the current cycle.
    # We try to remove it via a 2-opt move.
    
    while time.time() - start_time < time_limit:
        # Find the edge with the maximum index
        max_idx = -1
        max_pos = -1
        for i in range(N):
            u = cycle[i]
            v = cycle[(i + 1) % N]
            idx = get_edge_index(u, v)
            if idx > max_idx:
                max_idx = idx
                max_pos = i
        
        # Try all 2-opt moves that remove the edge at max_pos
        # The edge is (cycle[max_pos], cycle[max_pos+1])
        # A 2-opt move involving this edge would pair one of its endpoints with a vertex from a different edge.
        # Let edge A be (u, v) where u=cycle[max_pos], v=cycle[max_pos+1].
        # We can swap v with some other vertex w, effectively reversing a segment.
        
        # Simpler: Iterate over all other edges (k, k+1) and try swapping.
        found_improvement = False
        for j in range(N):
            if j == max_pos or j == (max_pos - 1) % N:
                continue
            
            # Edge 1: (u1, u2) = (cycle[max_pos], cycle[max_pos+1])
            # Edge 2: (v1, v2) = (cycle[j], cycle[j+1])
            
            # Two possible 2-opt connections:
            # Option 1: Connect u1-v1 and u2-v2. This reverses the segment between them.
            # Option 2: Connect u1-v2 and u2-v1. This also reverses a segment but differently.
            
            # Let's just perform the 2-opt move that removes both edges.
            # The new edges would be (cycle[max_pos], cycle[j]) and (cycle[max_pos+1], cycle[j+1])? 
            # No, standard 2-opt on a cycle:
            # Remove (a,b) and (c,d). Add (a,c) and (b,d).
            # Here a=cycle[max_pos], b=cycle[max_pos+1]
            # c=cycle[j], d=cycle[j+1]
            
            # We need to check if the new edges (a,c) and (b,d) are valid and what their indices are.
            # Also, the rest of the cycle remains the same.
            
            # To update the cycle array, we reverse the segment from b to c (or a to d).
            # Let's implement a helper to apply 2-opt and return new max.
            
            # For speed, we can estimate the new max.
            # The new max will be max(old_max_except_a_b_c_d, idx(a,c), idx(b,d)).
            # We don't know old_max_except... easily without recomputing.
            # Given N=24, recomputing max is O(N) = 24 ops. 
            # Number of moves per iteration: N. Total ops: 24*24 = 576. 
            # This is very fast.
            
            # Perform the move on a copy
            new_cycle = cycle[:]
            # Reverse segment from index max_pos+1 to j (inclusive)
            # This corresponds to removing (max_pos, max_pos+1) and (j, j+1) 
            # and adding (max_pos, j) and (max_pos+1, j+1)
            # Wait, reversing [max_pos+1 .. j] changes edges:
            # Old: ... a, b, ..., c, d, ...
            # New: ... a, c, ..., b, d, ...
            # Edges removed: (a,b) and (c,d).
            # Edges added: (a,c) and (b,d).
            
            if max_pos < j:
                new_cycle[max_pos+1:j+1] = reversed(new_cycle[max_pos+1:j+1])
            else:
                # Wrap around
                # This is tricky with list reversal. 
                # Easier: just construct the new cycle explicitly? 
                # Or handle wrap-around reversal.
                
                # Let's stick to non-wrapping for simplicity in the loop, 
                # but we must consider all j.
                # If max_pos > j, the segment wraps.
                # We can rotate the list so max_pos is 0? 
                # Or just implement generic 2-opt.
                
                # Let's use a simpler approach: 
                # The cycle is a sequence. 
                # We can just try all pairs of edges (i, j) and perform the 2-opt.
                
                # To avoid complex list manipulation errors, let's just recompute the cycle 
                # and max index for the candidate.
                # Since N is small, this is fine.
                
                pass # Placeholder for logic below
        
        # Let's refine the loop to just try all 2-opt moves and pick the best one.
        
        best_move = None
        best_new_max = current_max
        
        for j in range(N):
            if j == max_pos or j == (max_pos - 1) % N:
                continue
            
            # Define the two edges to be removed
            # e1 = (cycle[max_pos], cycle[max_pos+1])
            # e2 = (cycle[j], cycle[j+1])
            
            # The 2-opt move reverses the segment between the two edges.
            # We need to construct the new cycle.
            
            # To handle wrap-around uniformly, we can represent the cycle as a list 
            # starting at max_pos.
            
            # Actually, let's just do this:
            # Create a new list by taking the cycle, 
            # and reversing the appropriate slice.
            
            # If we remove edges (i, i+1) and (j, j+1), 
            # and assume i < j (in a linearized view), 
            # we reverse indices i+1 to j.
            
            # Let's linearize the cycle starting at max_pos.
            # This is getting complicated. 
            # Alternative: Just use a generic 2-opt implementation that works on indices.
            
            # Let's try a different move: Swap two vertices? 
            # No, 2-opt is better for TSP.
            
            # Let's implement 2-opt carefully.
            
            i1 = max_pos
            i2 = (max_pos + 1) % N
            j1 = j
            j2 = (j + 1) % N
            
            # The new edges will be (cycle[i1], cycle[j1]) and (cycle[i2], cycle[j2])
            # This is valid if the segment between i2 and j1 is reversed.
            
            # Let's build the new cycle:
            # Start at i1.
            # Go to j1.
            # Then traverse backwards from j1's predecessor in the reversed segment...
            
            # Simpler: 
            # The new cycle is:
            # [cycle[i1], cycle[j1], cycle[j1-1], ..., cycle[i2+1], cycle[i2], cycle[i2-1], ..., cycle[i1+1]]
            # Wait, this assumes i1 < j1 and no wrap.
            
            # Let's just use a helper function to apply 2-opt and return the new list.
            
            def apply_2_opt(c, i, j):
                # i and j are indices of the first vertex of the two edges to be cut.
                # Edge 1: c[i] - c[(i+1)%N]
                # Edge 2: c[j] - c[(j+1)%N]
                # Reverses segment from (i+1) to j.
                
                # To make it simple, rotate c so that i is at index 0.
                # This is O(N).
                # Or just construct the new list by iterating.
                
                new_c = [0] * N
                # The path goes from c[i] to c[j]
                # Then from c[j] it goes to c[j-1], c[j-2]... until c[i+1]
                # Then from c[i+1] it goes to c[i]
                # Wait, the other part of the cycle?
                # The cycle is closed.
                # The segment between i+1 and j is reversed.
                # The rest of the cycle (from j+1 to i-1) remains in the same order.
                
                # Let's trace:
                # Original: ... c[i], c[i+1], ..., c[j], c[j+1], ... c[i]
                # New: ... c[i], c[j], c[j-1], ..., c[i+1], c[j+1], ... c[i]
                
                # Indices in new_c:
                # new_c[0] = c[i]
                # new_c[1] = c[j]
                # new_c[2] = c[j-1]
                # ...
                # new_c[k] = c[j - (k-1)] for k=1..(j-i)
                # Then the rest of the vertices from j+1 to i-1 (wrapping) follow.
                
                # This is error-prone. Let's just use list slicing on a rotated list.
                
                # Rotate so i is at 0
                rotated = c[i:] + c[:i]
                # Now edge 1 is at 0-1.
                # We need to find the position of j in the rotated list.
                # j_rot = (j - i + N) % N
                j_rot = (j - i + N) % N
                
                # If j_rot == 0 or 1, skip (handled by caller)
                
                # Reverse segment 1 to j_rot
                # The edge to be cut is (rotated[0], rotated[1]) and (rotated[j_rot], rotated[j_rot+1])
                # Reversing rotated[1:j_rot+1] will make:
                # rotated[0] connected to rotated[j_rot]
                # and rotated[1] connected to rotated[j_rot+1]
                
                new_rotated = rotated[:]
                new_rotated[1:j_rot+1] = reversed(new_rotated[1:j_rot+1])
                
                # Rotate back
                # The new cycle starts at c[i] which is at index 0 of new_rotated.
                # So we just need to rotate new_rotated back by i.
                # But the "start" of the cycle doesn't matter for the set of edges.
                # However, the function returns a permutation.
                # We can return new_rotated as is, or rotate it.
                # The score depends on edges, not order.
                # But the interface expects a permutation.
                # Any rotation is a valid permutation describing the same cycle.
                
                return new_rotated

            new_cycle = apply_2_opt(cycle, max_pos, j)
            new_max = cycle_max_index(new_cycle)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_move = new_cycle
        
        if best_move:
            cycle = best_move
            current_max = best_new_max
            if current_max < best_max:
                best_max = current_max
                best_cycle = cycle[:]
        else:
            # No improvement found by removing the single worst edge.
            # Try random 2-opt moves that don't increase the max index?
            # Or just break and restart with a new random cycle.
            # Given the small N, random restarts are effective.
            break
    
    return best_cycle, best_max

def construct():
    # Run local search with multiple restarts
    best_cycle_overall = list(range(N))
    best_score_overall = cycle_max_index(best_cycle_overall)
    
    start_time = time.time()
    max_time = 55.0  # Leave 5s margin
    
    rng = random.Random(42)
    
    while time.time() - start_time < max_time:
        # Random initial cycle
        current_cycle = list(range(N))
        rng.shuffle(current_cycle)
        
        # Optimize
        opt_cycle, opt_score = optimize_cycle(current_cycle, time_limit=5.0)
        
        if opt_score < best_score_overall:
            best_score_overall = opt_score
            best_cycle_overall = opt_cycle
    
    return best_cycle_overall
# EVOLVE-BLOCK-END
