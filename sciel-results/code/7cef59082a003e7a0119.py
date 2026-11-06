import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Construct Grötzsch Graph on vertices 0..10
    # Cycle 0-1-2-3-4-0
    # Copies 5-9 where i+5 connected to N(i)
    # Apex 10 connected to 5-9
    edges = []
    # Cycle
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Mycielski edges
    for i in range(5):
        # i+5 connected to neighbors of i in cycle
        n1 = (i - 1) % 5
        n2 = (i + 1) % 5
        edges.append([5 + i, n1])
        edges.append([5 + i, n2])
    # Apex
    for i in range(5):
        edges.append([10, 5 + i])
    
    # Build adjacency matrix for G_11 (0..10)
    n_old = 11
    adj_old = [[False] * n_old for _ in range(n_old)]
    for u, v in edges:
        adj_old[u][v] = True
        adj_old[v][u] = True
        
    # Precompute all independent sets in G_11
    # We only care about "good" independent sets, likely maximal or large.
    # But for local search, we might need smaller ones.
    # Let's generate all independent sets up to size 6? Or just maximal ones?
    # Maximal independent sets are sufficient for "filling" a vertex's neighborhood if we don't need to intersect.
    # However, if we need disjoint sets, maximal might be too big.
    # Let's generate all independent sets of size >= 3.
    
    all_independent_sets = []
    # Use a simple backtracking to find independent sets
    def find_independent_sets():
        sets = []
        # Bitmask representation for speed? 11 bits.
        # Let's just use lists.
        
        # Recursive generation
        def backtrack(candidate, current):
            # Add current if size >= 3 (or 0)
            if len(current) >= 3:
                sets.append(list(current))
            
            if not candidate:
                return
            
            # Pick first vertex
            v = candidate[0]
            rest = candidate[1:]
            
            # Option 1: Include v
            # New candidate: vertices in rest not adjacent to v
            new_candidate = [u for u in rest if not adj_old[v][u]]
            backtrack(new_candidate, current + [v])
            
            # Option 2: Exclude v
            backtrack(rest, current)
        
        backtrack(list(range(n_old)), [])
        return sets

    # This might be slow if done naively, but n=11 is small.
    # Number of independent sets in Grötzsch is small.
    independent_sets = find_independent_sets()
    
    # Filter unique sets and sort by size descending
    unique_sets = [tuple(s) for s in independent_sets]
    unique_sets = list(set(unique_sets))
    unique_sets.sort(key=len, reverse=True)
    
    # If no sets found (shouldn't happen), fallback
    if not unique_sets:
        unique_sets = [()]
        
    # We will use a local search for the 5 new vertices (11..15)
    n_new = 5
    new_vertices = list(range(n_old, n_old + n_new))
    
    # State: 
    # neigh: list of sets (tuples) for each new vertex, representing neighbors in V_old
    # new_edges: set of edges between new vertices (represented as indices 0..4)
    
    def get_total_edges(neigh, new_edges):
        # Edges in G_11
        count = len(edges)
        # Edges from new to old
        for i in range(n_new):
            count += len(neigh[i])
        # Edges between new
        count += len(new_edges)
        return count

    def is_valid_config(neigh, new_edges):
        # 1. Each neigh[i] must be an independent set in G_11
        # We generate from independent_sets, so this is guaranteed if we only pick from there.
        # 2. If (i,j) in new_edges, then neigh[i] and neigh[j] must be disjoint
        for i, j in new_edges:
            if i > j: continue
            if neigh[i] & neigh[j]:
                return False
        # 3. New vertices themselves must form a triangle-free graph
        # Check for triangles in new_edges
        for u, v in new_edges:
            if v < u: continue
            for w in range(n_new):
                if w == u or w == v: continue
                if (v, w) in new_edges or (w, v) in new_edges:
                    # Check (u,v) and (v,w) and (u,w)
                    if (u, w) in new_edges or (w, u) in new_edges:
                        return False
        return True

    # Initialize: All new vertices connected to the largest independent set
    max_set = unique_sets[0]
    init_neigh = [max_set for _ in range(n_new)]
    init_new_edges = set()
    
    best_neigh = init_neigh
    best_new_edges = init_new_edges
    best_score = get_total_edges(best_neigh, best_new_edges)
    
    # Local Search
    # Moves:
    # 1. Change neigh[i] to another independent set
    # 2. Add an edge between new vertices i, j
    # 3. Remove an edge between new vertices i, j
    
    # To speed up, we might restrict candidate independent sets to top K largest
    # Since n_new is small, let's try top 50 sets
    candidate_sets = unique_sets[:50]
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Random restarts if stuck?
        # For now, simple hill climbing with random moves
        
        # Pick a random move
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Change neighbor set of a random new vertex
            i = random.randint(0, n_new - 1)
            if len(candidate_sets) > 1:
                new_set = random.choice(candidate_sets)
                if new_set == best_neigh[i]:
                    continue
                trial_neigh = list(best_neigh)
                trial_neigh[i] = new_set
                # Check validity
                # Only need to check disjointness for edges involving i
                valid = True
                for j in range(n_new):
                    if (i, j) in best_new_edges or (j, i) in best_new_edges:
                        if trial_neigh[i] & trial_neigh[j]:
                            valid = False
                            break
                if valid:
                    score = get_total_edges(trial_neigh, best_new_edges)
                    if score > best_score:
                        best_neigh = trial_neigh
                        best_score = score
        elif move_type == 1:
            # Add edge between new vertices
            i, j = random.sample(range(n_new), 2)
            if (i, j) in best_new_edges or (j, i) in best_new_edges:
                continue
            # Check if disjoint
            if best_neigh[i] & best_neigh[j]:
                continue
            # Check if adding edge creates triangle in new graph
            # Triangle (i, j, k) if (i,k) and (j,k) exist
            creates_triangle = False
            for k in range(n_new):
                if k == i or k == j: continue
                has_ik = (i, k) in best_new_edges or (k, i) in best_new_edges
                has_jk = (j, k) in best_new_edges or (k, j) in best_new_edges
                if has_ik and has_jk:
                    creates_triangle = True
                    break
            if not creates_triangle:
                trial_edges = set(best_new_edges)
                trial_edges.add((i, j))
                score = get_total_edges(best_neigh, trial_edges)
                if score > best_score:
                    best_new_edges = trial_edges
                    best_score = score
        elif move_type == 2:
            # Remove edge
            if not best_new_edges:
                continue
            i, j = random.choice(list(best_new_edges))
            trial_edges = set(best_new_edges)
            trial_edges.remove((i, j))
            # Removing an edge always keeps it valid and reduces score, so only do this if we want to enable other moves?
            # In simple hill climbing, we don't do this unless we allow down-moves.
            # Let's allow a small probability of down-moves (simulated annealing style) to escape local optima.
            if random.random() < 0.1:
                 score = get_total_edges(best_neigh, trial_edges)
                 # Accept with prob exp(-delta/T) ? T is high initially.
                 # For simplicity, just accept if random < 0.5 (downhill)
                 if random.random() < 0.5:
                     best_new_edges = trial_edges
                     best_score = score
                     
        # Occasionally restart with a different random configuration
        if iterations % 1000 == 0:
             # Pick random sets
             trial_neigh = [random.choice(candidate_sets) for _ in range(n_new)]
             trial_edges = set()
             # Ensure validity for edges (none initially)
             score = get_total_edges(trial_neigh, trial_edges)
             if score > best_score:
                 best_neigh = trial_neigh
                 best_new_edges = trial_edges
                 best_score = score

    # Construct final edge list
    final_edges = list(edges)
    
    # Add edges from new to old
    for i in range(n_new):
        v = n_old + i
        for u in best_neigh[i]:
            final_edges.append([v, u])
            
    # Add edges between new
    for i, j in best_new_edges:
        final_edges.append([n_old + i, n_old + j])
        
    return final_edges

