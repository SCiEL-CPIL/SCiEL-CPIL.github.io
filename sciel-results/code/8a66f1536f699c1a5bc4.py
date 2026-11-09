"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)

# Build edge index: for each (u,v) pair (u<v), store its arrival index (1-based)
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_idx[(a, b)] = i + 1

def get_edge_weight(u, v):
    a, b = min(u, v), max(u, v)
    return edge_idx.get((a, b), M + 1)

def cycle_score(cycle):
    """Compute the max arrival index over all edges in the Hamilton cycle."""
    n = len(cycle)
    max_arrival = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        w = get_edge_weight(u, v)
        if w > max_arrival:
            max_arrival = w
    return max_arrival

def greedy_construct():
    """Greedy: build a cycle by always picking the next vertex with the smallest
    edge arrival index from the current endpoint, ensuring we can still complete."""
    # We'll build a path, then close it.
    # Start from vertex 0.
    used = set()
    path = [0]
    used.add(0)
    
    for step in range(1, N):
        current = path[-1]
        best_next = -1
        best_w = M + 1
        # Try to find next vertex with smallest edge weight,
        # but avoid trapping ourselves (unless it's the last step)
        candidates = []
        for v in range(N):
            if v not in used:
                w = get_edge_weight(current, v)
                candidates.append((w, v))
        
        if not candidates:
            break
        
        # Sort by weight
        candidates.sort()
        
        # For the last step, we must be able to close back to 0
        if step == N - 1:
            for w, v in candidates:
                # Check if edge v->0 exists and is reasonable
                w_close = get_edge_weight(v, 0)
                if w_close <= M:
                    best_next = v
                    break
            if best_next == -1:
                # Fallback: just pick the best
                best_next = candidates[0][1]
        else:
            # Pick the best, but check it doesn't create a dead end
            for w, v in candidates:
                # Heuristic: check that v has at least one unused neighbor (besides path[0] for closing)
                has_option = False
                for v2 in range(N):
                    if v2 not in used and v2 != v:
                        if get_edge_weight(v, v2) <= M:
                            has_option = True
                            break
                # Also check closing edge to 0
                if get_edge_weight(v, 0) <= M:
                    has_option = True
                
                if has_option or step >= N - 2:
                    best_next = v
                    break
            if best_next == -1:
                best_next = candidates[0][1]
        
        path.append(best_next)
        used.add(best_next)
    
    return path

def two_opt_improve(cycle, max_iterations=50000, deadline=None):
    """2-opt local search: try all pairs of edges to reverse a segment."""
    n = len(cycle)
    best_cycle = list(cycle)
    best_score = cycle_score(best_cycle)
    
    # Precompute edge weights for the cycle
    def compute_score(cyc):
        mx = 0
        for i in range(n):
            w = get_edge_weight(cyc[i], cyc[(i+1) % n])
            if w > mx:
                mx = w
        return mx
    
    improved = True
    iters = 0
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        if deadline and time.time() > deadline:
            break
        
        for i in range(n):
            if deadline and time.time() > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # This would just reverse the whole cycle
                # Try reversing segment from i+1 to j
                # Current edges: (i, i+1) and (j, j+1)
                # New edges: (i, j) and (i+1, j+1)
                u1, v1 = best_cycle[i], best_cycle[i+1]
                u2, v2 = best_cycle[j], best_cycle[(j+1) % n]
                
                w_old1 = get_edge_weight(u1, v1)
                w_old2 = get_edge_weight(u2, v2)
                w_new1 = get_edge_weight(u1, u2)
                w_new2 = get_edge_weight(v1, v2)
                
                old_max = max(w_old1, w_old2)
                new_max = max(w_new1, w_new2)
                
                if new_max < old_max:
                    # Apply the 2-opt move
                    best_cycle[i+1:j+1] = reversed(best_cycle[i+1:j+1])
                    best_score = compute_score(best_cycle)
                    improved = True
                    break
            if improved:
                break
    
    return best_cycle, best_score

def random_restart_construct():
    """Create a random Hamilton cycle."""
    cycle = list(range(N))
    random.shuffle(cycle)
    return cycle

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin: 80s out of 90s limit
    
    # Phase 1: Greedy construction
    best_cycle = greedy_construct()
    best_score = cycle_score(best_cycle)
    
    # Phase 2: 2-opt improvement
    best_cycle, best_score = two_opt_improve(best_cycle, max_iterations=100000, deadline=deadline)
    
    # Phase 3: Random restarts with 2-opt
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        cand = random_restart_construct()
        cand, cand_score = two_opt_improve(cand, max_iterations=20000, deadline=deadline)
        if cand_score < best_score:
            best_score = cand_score
            best_cycle = cand
        # Early stop if we've found a very good solution
        if best_score <= 50:
            break
    
    # Phase 4: Final polish with more iterations
    if time.time() < deadline - 2:
        best_cycle, best_score = two_opt_improve(best_cycle, max_iterations=200000, deadline=deadline)
    
    return best_cycle

# EVOLVE-BLOCK-END
