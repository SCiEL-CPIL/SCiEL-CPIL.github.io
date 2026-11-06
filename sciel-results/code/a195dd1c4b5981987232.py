import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    best_path = []
    best_len = 0
    
    # Phase 1: Fast greedy backtracking to establish a strong baseline
    def try_search(start_v, time_limit):
        used = {start_v}
        forbidden = set()
        path = [start_v]
        
        while time.time() < time_limit:
            cur = path[-1]
            neighbors = NEIGHBORS[cur]
            
            candidates = []
            for u in neighbors:
                if u in used or u in forbidden:
                    continue
                candidates.append(u)
            
            if not candidates:
                break
            
            if len(candidates) == 1:
                next_v = candidates[0]
            else:
                # Heuristic: pick the candidate that leaves the most options for the next step
                best_score = -1
                best_v = None
                for u in candidates:
                    score = 0
                    for w in NEIGHBORS[u]:
                        if w in used: continue
                        if w in forbidden: continue
                        if w in NEIGHBORS[cur]: continue # Will be forbidden
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_v = u
                    elif score == best_score and random.random() < 0.5:
                        best_v = u
                
                next_v = best_v
            
            path.append(next_v)
            used.add(next_v)
            
            # Update forbidden: add neighbors of the vertex we just left (cur)
            for n in NEIGHBORS[cur]:
                forbidden.add(n)
        
        return path

    # Run a few quick greedy searches to get a baseline
    baseline_deadline = start_time + 10.0
    for _ in range(5):
        if time.time() >= baseline_deadline:
            break
        start_v = random.randint(0, SIZE - 1)
        p = try_search(start_v, baseline_deadline)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 2: MILP formulation for maximum induced path
    # Using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        milp_start = time.time()
        milp_deadline = start_time + 110.0  # Leave 5s for final verification
        
        # MILP variables:
        # x[v] = 1 if vertex v is in the path
        # y[u,v] = 1 if edge (u,v) is used in the path (u < v)
        # t[v] = position of v in the path (for subtour elimination)
        
        # Number of edges in hypercube
        num_edges = SIZE * N // 2
        edge_list = []
        for u in range(SIZE):
            for b in range(N):
                v = u ^ (1 << b)
                if u < v:
                    edge_list.append((u, v))
        
        # Variable indices:
        # 0 to SIZE-1: x[v]
        # SIZE to SIZE + num_edges - 1: y[u,v]
        # SIZE + num_edges to SIZE + num_edges + SIZE - 1: t[v]
        
        n_vars = SIZE + num_edges + SIZE
        
        # Objective: maximize sum of x[v] (number of vertices in path)
        # Or equivalently, maximize sum of y[u,v] + 1
        c = np.zeros(n_vars)
        for v in range(SIZE):
            c[v] = -1  # Minimize negative = maximize
        
        # Constraints:
        # 1. Degree constraint: each vertex has degree <= 2 in the path
        #    For vertex v: sum of y[u,v] over all neighbors u <= 2
        # 2. Subtour elimination: MTZ constraints
        #    If edge (u,v) is used, then t[u] + 1 <= t[v] or t[v] + 1 <= t[u]
        #    This is tricky with binary y variables.
        
        # Alternative: use the fact that an induced path has no chords.
        # For non-consecutive vertices u, v: they cannot be adjacent.
        # This is hard to encode directly.
        
        # Simpler approach: use the degree constraint + connectivity.
        # But ensuring it's a single path (not multiple paths) is hard.
        
        # Let's try a different MILP formulation:
        # x[v] = 1 if v is in path
        # y[u,v] = 1 if edge (u,v) is in path
        # 
        # Constraints:
        # 1. For each vertex v: sum_{u adj to v} y[u,v] <= 2
        # 2. For each vertex v: x[v] >= sum_{u adj to v} y[u,v] / 2
        # 3. For each vertex v: x[v] <= sum_{u adj to v} y[u,v] + 1 (if degree > 0)
        # 4. No triangles: for any triangle (u,v,w), y[u,v] + y[v,w] + y[w,u] <= 2
        #    But hypercube has no triangles, so this is automatic.
        # 5. Induced path: for any edge (u,v) not in path, if both u and v are in path,
        #    then they must be consecutive. This is hard.
        
        # Given the complexity, let's use a simpler heuristic-based MILP:
        # Maximize path length subject to degree <= 2 and no chords.
        
        # Actually, let's just use the greedy result and try to improve it with local search.
        # The MILP approach is too complex for the time limit.
        
        # Instead, do more greedy searches with different heuristics.
        
    except Exception as e:
        pass
    
    # Phase 3: More greedy searches with varied heuristics
    search_deadline = start_time + 110.0
    attempts = 0
    while time.time() < search_deadline:
        attempts += 1
        start_v = random.randint(0, SIZE - 1)
        p = try_search(start_v, search_deadline)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
        # Occasionally try structured starts
        if attempts % 10 == 0:
            for sv in [0, SIZE-1, 1, SIZE-2]:
                p = try_search(sv, search_deadline)
                if len(p) > best_len:
                    best_len = len(p)
                    best_path = p
    
    # Phase 4: Try to extend the best path found
    if best_path and time.time() < start_time + 112:
        # Try extending from both ends
        def try_extend(path, end_is_front):
            if end_is_front:
                # Extend from front
                cur = path[0]
                used = set(path)
                forbidden = set()
                # Compute forbidden for front extension
                for v in path[1:]:
                    for n in NEIGHBORS[v]:
                        forbidden.add(n)
                
                while time.time() < start_time + 112:
                    candidates = [u for u in NEIGHBORS[cur] if u not in used and u not in forbidden]
                    if not candidates:
                        break
                    # Pick best candidate
                    best_score = -1
                    best_v = None
                    for u in candidates:
                        score = 0
                        for w in NEIGHBORS[u]:
                            if w in used: continue
                            if w in forbidden: continue
                            if w in NEIGHBORS[cur]: continue
                            score += 1
                        if score > best_score:
                            best_score = score
                            best_v = u
                    path.insert(0, best_v)
                    used.add(best_v)
                    for n in NEIGHBORS[cur]:
                        forbidden.add(n)
                    cur = best_v
            else:
                # Extend from back (standard)
                used = set(path)
                forbidden = set()
                for v in path[:-1]:
                    for n in NEIGHBORS[v]:
                        forbidden.add(n)
                
                while time.time() < start_time + 112:
                    cur = path[-1]
                    candidates = [u for u in NEIGHBORS[cur] if u not in used and u not in forbidden]
                    if not candidates:
                        break
                    best_score = -1
                    best_v = None
                    for u in candidates:
                        score = 0
                        for w in NEIGHBORS[u]:
                            if w in used: continue
                            if w in forbidden: continue
                            if w in NEIGHBORS[cur]: continue
                            score += 1
                        if score > best_score:
                            best_score = score
                            best_v = u
                    path.append(best_v)
                    used.add(best_v)
                    for n in NEIGHBORS[cur]:
                        forbidden.add(n)
            
            return path
        
        extended = try_extend(best_path[:], False)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        
        extended = try_extend(best_path[:], True)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    # Final verification
    if best_path and len(best_path) > 1:
        for i in range(len(best_path)-1):
            if (best_path[i] ^ best_path[i+1]).bit_count() != 1:
                best_path = [0]
                break
        else:
            for i in range(len(best_path)):
                for j in range(i+2, len(best_path)):
                    if (best_path[i] ^ best_path[j]).bit_count() == 1:
                        best_path = [0]
                        break
                if len(best_path) == 1:
                    break
    
    return best_path if best_path else [0]

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)}")
