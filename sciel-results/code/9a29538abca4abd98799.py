"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = 24
PATH_SET = set((min(u, v), max(u, v)) for u, v in P['edges'])

def _build_adj(added):
    adj = [[] for _ in range(N)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_valid(added):
    # Check for duplicates and path edges
    edge_set = set(PATH_SET)
    for u, v in added:
        k = (min(u, v), max(u, v))
        if k in edge_set:
            return False
        edge_set.add(k)
    
    adj = _build_adj(added)
    
    # Triangle-free check
    adj_set = [set(a) for a in adj]
    for u in range(N):
        for v in adj[u]:
            if v > u: # only check once per edge
                # If v and u have a common neighbor, triangle exists
                # Actually, triangle is u-v-w.
                # For each edge (u,v), check if any w is neighbor to both.
                # More efficient: for each u, check if neighbors form a clique (size > 1 implies triangle in bipartite? No).
                # In bipartite graph, no triangles.
                # We only add edges between even and odd (if we enforce that).
                # But _check_valid should be general.
                pass
    
    # General triangle check
    for u in range(N):
        nbrs = adj[u]
        nbr_set = set(nbrs)
        for v in nbrs:
            for w in adj[v]:
                if w in nbr_set and w != u:
                    return False

    # Diameter check
    for src in range(N):
        dist = [-1] * N
        dist[src] = 0
        q = deque([src])
        while q:
            node = q.popleft()
            d = dist[node]
            if d >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    if dist[nb] > 4:
                        return False
                    q.append(nb)
        if -1 in dist:
            return False
            
    return True

def _get_diameter(added):
    edge_set = set(PATH_SET)
    for u, v in added:
        k = (min(u, v), max(u, v))
        if k in edge_set:
            return 100 # invalid
        edge_set.add(k)
        
    adj = _build_adj(added)
    
    # Triangle check
    for u in range(N):
        nbrs = adj[u]
        nbr_set = set(nbrs)
        for v in nbrs:
            for w in adj[v]:
                if w in nbr_set and w != u:
                    return 100

    max_d = 0
    for src in range(N):
        dist = [-1] * N
        dist[src] = 0
        q = deque([src])
        while q:
            node = q.popleft()
            d = dist[node]
            if d > max_d:
                max_d = d
            if d >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    q.append(nb)
        if -1 in dist:
            return 100
    return max_d

def _greedy_construct():
    """Construct a valid solution using greedy addition."""
    added = []
    added_set = set()
    # Candidates: all even-odd pairs not in path
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u % 2) != (v % 2) and (u, v) not in PATH_SET:
                candidates.append((u, v))
    
    # Shuffle for randomness? No, deterministic first.
    # Sort by "potential" - e.g., distance in path?
    # Large distance edges are better for diameter reduction.
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    while True:
        d = _get_diameter(added)
        if d <= 4:
            break
        if d == 100: # Invalid state, shouldn't happen if we only add valid edges
            break
            
        # Find the best edge to add
        best_edge = None
        best_d = d
        for u, v in candidates:
            if (u, v) in added_set:
                continue
            # Quick triangle check: u and v must not have common neighbor
            # Since we only add even-odd, and path is bipartite, 
            # a triangle would require a path of length 2 between u and v.
            # i.e., dist_path(u, v) == 2.
            # If dist_path(u, v) == 2, they share a neighbor. Adding edge creates triangle.
            if abs(u - v) == 2:
                continue
            
            trial = added + [[u, v]]
            td = _get_diameter(trial)
            if td < best_d:
                best_d = td
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        added.append(list(best_edge))
        added_set.add(best_edge)
        
    return added

def _local_search(added, time_limit=10.0):
    """Local search to minimize edge count."""
    start_time = time.time()
    cur = [list(e) for e in added]
    cur_set = set((min(u, v), max(u, v)) for u, v in cur)
    
    # Candidates for swap: all even-odd non-path edges
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u % 2) != (v % 2) and (u, v) not in PATH_SET:
                candidates.append((u, v))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Try removals
        for i in range(len(cur)):
            if (time.time() - start_time) > time_limit:
                break
            trial = cur[:i] + cur[i+1:]
            if _check_valid(trial):
                cur = trial
                cur_set.remove(tuple(cur[i])) # careful with indices
                # Recalculate set
                cur_set = set((min(u, v), max(u, v)) for u, v in cur)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try swaps (1-for-1) to enable future removals
        # Only try if we are stuck
        for i in range(len(cur)):
            if (time.time() - start_time) > time_limit:
                break
            rem_edge = cur[i]
            rem_key = (min(rem_edge[0], rem_edge[1]), max(rem_edge[0], rem_edge[1]))
            remaining = cur[:i] + cur[i+1:]
            
            for cand in candidates:
                if cand == rem_key:
                    continue
                if cand in cur_set:
                    continue
                # Quick check: cand must not create triangle
                # We check validity of the whole set
                trial = remaining + [list(cand)]
                if _check_valid(trial):
                    # Accept swap if it doesn't increase size (it doesn't)
                    # We accept it to potentially allow removals later
                    cur = trial
                    cur_set = set((min(u, v), max(u, v)) for u, v in cur)
                    improved = True
                    break
            if improved:
                break
                
    return cur

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    # Phase 1: Construct initial solution
    best = _greedy_construct()
    
    # Phase 2: Local Search
    best = _local_search(best, time_limit=10.0)
    
    # Phase 3: Random Restarts if time permits
    rng = random.Random(42)
    best_len = len(best)
    
    while (time.time() - start_time) < deadline:
        # Generate a random initial solution
        # Shuffle candidates
        candidates = []
        for u in range(N):
            for v in range(u + 1, N):
                if (u % 2) != (v % 2) and (u, v) not in PATH_SET:
                    candidates.append((u, v))
        rng.shuffle(candidates)
        
        # Greedy add with random order
        added = []
        added_set = set()
        while True:
            d = _get_diameter(added)
            if d <= 4:
                break
            if d == 100:
                break
            best_edge = None
            best_d = d
            # Limit search to top 20 candidates by distance to save time
            for u, v in candidates:
                if (u, v) in added_set:
                    continue
                if abs(u - v) == 2:
                    continue
                trial = added + [[u, v]]
                td = _get_diameter(trial)
                if td < best_d:
                    best_d = td
                    best_edge = (u, v)
            if best_edge is None:
                break
            added.append(list(best_edge))
            added_set.add(best_edge)
            
        # Local search on this new solution
        trial_best = _local_search(added, time_limit=5.0)
        
        if len(trial_best) < best_len:
            best = trial_best
            best_len = len(best)
            if best_len <= 4: # Theoretical lower bound guess
                break

    # Ensure types
    result = [[int(e[0]), int(e[1])] for e in best]
    return result

# EVOLVE-BLOCK-END
