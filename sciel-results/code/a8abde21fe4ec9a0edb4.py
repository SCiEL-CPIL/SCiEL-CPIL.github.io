import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit (evaluator kills at 90s, but task says 60s in contract? "Each evaluation must finish within 60 s" is in the text. Wait, "evaluator KILLS the program after 90s wall-clock" in directive. I will use 55s to be safe for the 60s constraint mentioned in the task contract, or 85s if the 90s is the hard limit. The task contract says "Each evaluation must finish within 60 s". The directive says "evaluator KILLS... after 90s". I will aim for < 55s to be safe against the 60s contract constraint, as exceeding 60s might score 0 even if not killed at 90s.)
    
    N = 256
    # Vertices are 0..N-1 representing values 1..N
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency list for conflict graph
    # Edge (i, j) exists if (i+1) + (j+1) is a perfect square.
    # Note: The problem says "a+b is never a perfect square for a<=b in the set, INCLUDING a=b".
    # This means:
    # 1. If a != b, a+b must not be a square.
    # 2. If a == b, 2a must not be a square.
    # So, vertex i is "self-conflicting" if 2*(i+1) is a square. Such vertices cannot be in the set.
    # For distinct i, j, they conflict if (i+1)+(j+1) is a square.
    
    adjacency = [set() for _ in range(N)]
    self_conflict = [False] * N
    
    for i in range(N):
        val_i = i + 1
        # Check self-conflict
        if 2 * val_i in squares:
            self_conflict[i] = True
            continue # If self-conflicting, it can't be in any valid set, so no need to add edges? 
            # Actually, we can just exclude it from the candidate pool entirely.
        
        for j in range(i + 1, N):
            val_j = j + 1
            if (val_i + val_j) in squares:
                adjacency[i].add(j)
                adjacency[j].add(i)
    
    # Filter out self-conflicting vertices
    valid_vertices = [i for i in range(N) if not self_conflict[i]]
    
    # Map valid vertices to 0..M-1 indices for the MIS solver
    M = len(valid_vertices)
    if M == 0:
        return [1] # Fallback, though there should be valid vertices
    
    # Re-index adjacency for the subgraph of valid vertices
    # original_index -> new_index
    idx_map = {orig: new for new, orig in enumerate(valid_vertices)}
    
    adj = [set() for _ in range(M)]
    for orig_i in valid_vertices:
        new_i = idx_map[orig_i]
        for orig_j in adjacency[orig_i]:
            if orig_j in idx_map:
                new_j = idx_map[orig_j]
                adj[new_i].add(new_j)
    
    # Tomita's Algorithm for Maximum Independent Set
    # We want to find the largest subset of vertices with no edges between them.
    
    best_size = 0
    best_set = []
    
    # Sort vertices by degree descending to help pruning? 
    # Standard Tomita uses a specific ordering. Let's use a simple branch and bound.
    
    # To make it fast, we use a recursive DFS with coloring bound (Tomita).
    # However, implementing full Tomita with coloring is complex. 
    # A simpler Branch and Bound with greedy coloring upper bound is often sufficient for N=256 if the graph is sparse.
    # The conflict graph here is quite sparse. 
    # Squares up to 512: 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484.
    # Each number x conflicts with s-x for each square s.
    # Max degree is small (around 5-10).
    # With small degree, a simple backtracking with pruning should work very fast.
    
    # Let's implement a standard recursive MIS solver.
    
    def solve_mis():
        nonlocal best_size, best_set
        
        # Order vertices by degree (descending) to improve pruning
        # We'll work with a list of candidate vertices.
        # We need to maintain the current independent set and the candidates.
        
        # A more efficient approach for sparse graphs:
        # Use a recursive function that takes a set of candidates and current set size.
        # Upper bound: len(candidates) + current_size.
        # If upper bound <= best_size, prune.
        
        # To be even faster, we can use a greedy coloring bound.
        # Let's try a simpler branch and bound first. If it's too slow, we'll add coloring.
        
        # Since M is ~250 and degree is low, we can use a simple DFS.
        
        # Convert adj to list of lists for faster iteration
        adj_list = [list(s) for s in adj]
        
        # Sort vertices by degree descending initially? 
        # Actually, dynamic ordering is better, but static is okay for start.
        
        def dfs(candidates, current_set):
            nonlocal best_size, best_set
            
            if not candidates:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return
            
            # Pruning: even if we add all candidates, can we beat best?
            if len(current_set) + len(candidates) <= best_size:
                return
            
            # Heuristic: pick a vertex with highest degree in the candidate subgraph?
            # Or just pick the first one? 
            # Picking a high-degree vertex often prunes more branches.
            
            # Let's compute degrees in the induced subgraph of candidates
            # This is O(N^2) per step which is too slow if done naively.
            # Instead, let's just pick the vertex with the highest original degree among candidates.
            
            # Optimization: If the number of candidates is small, we can be exhaustive.
            
            # Pick a vertex v from candidates
            # Let's pick v with max degree in the *original* graph to start with, 
            # but really we should pick based on the induced subgraph.
            
            # For simplicity and speed, let's pick the first vertex in the list, 
            # but sort the initial list by degree descending.
            
            # Let's implement a better heuristic: 
            # Choose v in candidates with the highest number of neighbors in candidates.
            
            # To avoid O(N^2) calculation every step, we can approximate or just pick one.
            # Given the low density, picking any vertex is okay, but high-degree is better.
            
            # Let's just pick the first vertex in the sorted list.
            # We will maintain candidates as a list.
            
            v = candidates[0]
            remaining = candidates[1:]
            
            # Branch 1: Include v
            # New candidates are those in remaining that are NOT neighbors of v
            new_candidates = [u for u in remaining if u not in adj[v]]
            current_set.append(v)
            dfs(new_candidates, current_set)
            current_set.pop()
            
            # Branch 2: Exclude v
            dfs(remaining, current_set)
            
        # Initial candidate list: all valid vertices
        # Sort by degree descending
        candidates = sorted(range(M), key=lambda x: len(adj[x]), reverse=True)
        
        # We can add a greedy initial solution to set a good lower bound
        # Greedy MIS: repeatedly pick vertex with min degree, add to set, remove it and its neighbors
        greedy_set = []
        remaining_verts = set(range(M))
        while remaining_verts:
            # Pick vertex with min degree in remaining subgraph
            min_deg = float('inf')
            min_v = None
            for v in remaining_verts:
                deg = 0
                for u in adj[v]:
                    if u in remaining_verts:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            
            greedy_set.append(min_v)
            # Remove min_v and its neighbors from remaining
            to_remove = {min_v} | adj[min_v]
            remaining_verts -= to_remove
            
        best_size = len(greedy_set)
        best_set = list(greedy_set)
        
        # Now run the DFS to improve
        # To prevent stack overflow or excessive time, we rely on the pruning.
        # For N=256, this should be fast if the graph is sparse.
        
        # One optimization: if best_size == M, we are done.
        if best_size < M:
            dfs(candidates, [])
            
        return best_set, best_size
    
    final_set_indices, final_size = solve_mis()
    
    # Map back to 1-based values
    result = [valid_vertices[i] + 1 for i in final_set_indices]
    result.sort()
    
    # Verify solution
    def is_valid_solution(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a <= b:
                    if (a + b) in squares:
                        return False
        return True
        
    if not is_valid_solution(result):
        # Fallback to a simple valid set
        # Pick numbers that are far apart? 
        # Just return [1] as safe fallback
        result = [1]
        
    return result

