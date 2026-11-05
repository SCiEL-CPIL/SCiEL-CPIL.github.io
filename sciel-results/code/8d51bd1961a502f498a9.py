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

def optimize_cycle(cycle, time_limit=1.0):
    start_time = time.time()
    current_max = cycle_max_index(cycle)
    best_cycle = cycle[:]
    best_max = current_max
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Standard 2-opt: try all pairs (i, j)
        # To be efficient, we can iterate and break early if we find an improvement,
        # but for bottleneck TSP, steepest descent might be better.
        # Let's find the best move in a full pass.
        best_move = None
        best_new_max = current_max
        
        for i in range(N):
            # Optimization: if time is up, break
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 2, N):
                if j == (i - 1 + N) % N: # Adjacent edges
                    continue
                
                # Calculate new max for 2-opt move (i, j)
                # Edges removed: (cycle[i], cycle[i+1]) and (cycle[j], cycle[j+1])
                # Edges added: (cycle[i], cycle[j]) and (cycle[i+1], cycle[j+1])
                # Note: This assumes we reverse the segment between i+1 and j.
                # Let's verify the connectivity.
                # Original: ... A(i) - B(i+1) ... C(j) - D(j+1) ... A
                # New: ... A - C ... B - D ... A
                # The segment B...C is reversed.
                # The edges inside the segment remain the same set.
                # So new_max = max( max(edges_not_involving_i_or_j), idx(A,C), idx(B,D) )
                
                # For N=24, recomputing full max is O(24).
                # Let's just build the new cycle and compute max.
                # This is simpler and less error-prone.
                
                # Construct new cycle
                # Reverse cycle[i+1 : j+1]
                # If i < j:
                # new_cycle = cycle[:i+1] + reversed(cycle[i+1:j+1]) + cycle[j+1:]
                
                new_cycle = cycle[:]
                new_cycle[i+1:j+1] = reversed(new_cycle[i+1:j+1])
                
                new_max = cycle_max_index(new_cycle)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = new_cycle
        
        if best_move:
            cycle = best_move
            current_max = best_new_max
            improved = True
            if current_max < best_max:
                best_max = current_max
                best_cycle = cycle[:]
                
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
        opt_cycle, opt_score = optimize_cycle(current_cycle, time_limit=1.0)
        
        if opt_score < best_score_overall:
            best_score_overall = opt_score
            best_cycle_overall = opt_cycle[:]
            
        # If we found a very good solution, we can try to polish it more
        if best_score_overall <= 100: # Heuristic threshold
             # Try one more polish on the best cycle found so far
             pol_cycle, pol_score = optimize_cycle(best_cycle_overall, time_limit=2.0)
             if pol_score < best_score_overall:
                 best_score_overall = pol_score
                 best_cycle_overall = pol_cycle[:]

    return best_cycle_overall
# EVOLVE-BLOCK-END
