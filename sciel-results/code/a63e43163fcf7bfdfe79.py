"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = 24

def _path_edges_set():
    s = set()
    for u, v in P['edges']:
        s.add((min(u,v), max(u,v)))
    return s

PATH_SET = _path_edges_set()

def _all_non_path_edges():
    """All candidate edges not in the path."""
    out = []
    for u in range(N):
        for v in range(u+1, N):
            if (u, v) not in PATH_SET:
                out.append((u, v))
    return out

ALL_NONPATH = _all_non_path_edges()

def _build_adj(added):
    adj = [[] for _ in range(N)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_diameter(adj):
    maxd = 0
    for s in range(N):
        dist = [-1]*N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= maxd:
                pass
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if dist[v] > maxd:
                        maxd = dist[v]
                    q.append(v)
        if -1 in dist:
            return -1
    return maxd

def _triangle_free(adj):
    adj_set = [set(a) for a in adj]
    for u in range(N):
        for v in adj[u]:
            for w in adj[u]:
                if v < w and w in adj_set[v]:
                    return False
    return True

def _quick_valid(added):
    es = set(PATH_SET)
    for u, v in added:
        k = (min(u,v), max(u,v))
        if k in es:
            return False
        es.add(k)
    adj = _build_adj(added)
    adj_set = [set(a) for a in adj]
    for u in range(N):
        for v in adj[u]:
            for w in adj[u]:
                if v < w and w in adj_set[v]:
                    return False
    # diameter via BFS with early stop
    for s in range(N):
        dist = [-1]*N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du > 4:
                return False
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if dist[v] > 4:
                        return False
                    q.append(v)
        if -1 in dist:
            return False
    return True

def _initial_solution():
    # 10-edge star at vertex 12 to odd vertices on each side
    return [[12,1],[12,3],[12,5],[12,7],[12,9],
            [12,15],[12,17],[12,19],[12,21],[12,23]]

def _greedy_removal(start_edges, max_time=10):
    """Start from a valid set; repeatedly try removing any edge that keeps validity."""
    start_time = time.time()
    cur = [list(e) for e in start_edges]
    if not _quick_valid(cur):
        return start_edges
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(len(cur)):
            if time.time() - start_time > max_time:
                break
            trial = cur[:i] + cur[i+1:]
            if _quick_valid(trial):
                cur = trial
                improved = True
                break
    return cur

def _search_for_9(max_time=30):
    """Try to find a 9-edge solution using randomized local search."""
    start_time = time.time()
    deadline = start_time + max_time
    rng = random.Random(42)
    best = None
    
    # Candidate pool: edges likely to be useful (long range)
    # We focus on edges that connect distant vertices
    candidates = [e for e in ALL_NONPATH if abs(e[0]-e[1]) >= 3]
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Randomly pick 9 edges from candidates
        # Bias towards hub-like structures?
        # Just random sampling for now
        if len(candidates) < 9:
            break
            
        trial_edges = rng.sample(candidates, k=9)
        trial = [list(e) for e in trial_edges]
        
        if _quick_valid(trial):
            return trial
            
        # If not valid, try to fix by swapping
        # This is expensive, so we just do a few iterations of swap
        for _ in range(10):
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            if not trial:
                continue
            idx = rng.randint(0, len(trial)-1)
            removed = trial.pop(idx)
            
            # Pick a random edge to add
            # Prefer edges not in trial
            existing_set = set((min(u,v),max(u,v)) for u,v in trial)
            existing_set |= PATH_SET
            valid_adds = [e for e in candidates if (min(e[0],e[1]),max(e[0],e[1])) not in existing_set]
            if not valid_adds:
                trial.append(removed)
                continue
                
            new_edge = rng.choice(valid_adds)
            trial.append(list(new_edge))
            
            if _quick_valid(trial):
                return trial
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    # Start with known 10-edge solution
    solution = _initial_solution()
    
    if not _quick_valid(solution):
        # Fallback: should not happen
        return []

    # Try to find a 9-edge solution
    # This is the main improvement attempt
    if time.time() < deadline - 5:
        remaining = min(30, deadline - time.time() - 5)
        if remaining > 0:
            sol9 = _search_for_9(max_time=remaining)
            if sol9 is not None and len(sol9) < len(solution):
                solution = sol9

    # If we found a 9-edge solution, try to remove edges (unlikely to work, but check)
    # Or if we are at 10, try to remove (we know 10 is minimal in this family, but maybe a different 10 is easier to reduce?)
    # Actually, just run greedy removal on the current best
    if time.time() < deadline - 2:
        remaining = min(10, deadline - time.time() - 2)
        if remaining > 0:
            solution = _greedy_removal(solution, max_time=remaining)

    # Final verification
    if not _quick_valid(solution):
        solution = _initial_solution()

    # Ensure return type is correct
    result = []
    for e in solution:
        result.append([int(e[0]), int(e[1])])

    return result
# EVOLVE-BLOCK-END
