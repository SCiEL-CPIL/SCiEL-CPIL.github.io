"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE = set()
for e in P['edges']:
    u, v = e
    BASE.add((min(u,v), max(u,v)))

def build_adj(added):
    adj = [set() for _ in range(N)]
    for (u,v) in BASE:
        adj[u].add(v); adj[v].add(u)
    for (u,v) in added:
        adj[u].add(v); adj[v].add(u)
    return adj

def check_solution(added):
    """Return (valid, bad_count). bad_count = number of pairs with dist>4 if valid, -1 if invalid."""
    adj = build_adj(added)
    # Check connected
    seen = [False]*N
    dq = deque([0])
    seen[0] = True
    cnt = 1
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                cnt += 1
                dq.append(v)
    if cnt != N:
        return (False, -1)
    # Check triangle-free
    for u in range(N):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return (False, -1)
    # Compute bad pairs
    bad = 0
    for s in range(N):
        dist = [-1]*N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u]+1
                    dq.append(v)
        for v in range(s+1, N):
            if dist[v] > 4:
                bad += 1
    return (True, bad)

def all_candidates():
    cands = []
    for u in range(N):
        for v in range(u+1, N):
            if (u,v) not in BASE:
                cands.append((u,v))
    return cands

def solve():
    deadline = time.time() + 80
    cands = all_candidates()
    best_solution = None
    best_count = float('inf')
    
    # Greedy
    added = []
    added_set = set()
    valid, bad_count = check_solution(added)
    if not valid:
        return []
    
    while bad_count > 0 and time.time() < deadline:
        best_edge = None
        best_bad = bad_count
        for e in cands:
            if e in added_set:
                continue
            trial = added + [e]
            valid2, bad2 = check_solution(trial)
            if valid2 and bad2 < best_bad:
                best_bad = bad2
                best_edge = e
                if best_bad == 0:
                    break
        if best_edge is None:
            break
        added.append(best_edge)
        added_set.add(best_edge)
        bad_count = best_bad
    
    if bad_count == 0:
        best_solution = list(added)
        best_count = len(added)
    
    # Local search: removal
    if best_solution is not None:
        cur = list(best_solution)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(cur)):
                if time.time() >= deadline:
                    break
                trial = cur[:i] + cur[i+1:]
                valid2, bad2 = check_solution(trial)
                if valid2 and bad2 == 0:
                    cur = trial
                    improved = True
                    break
        if len(cur) < best_count:
            best_count = len(cur)
            best_solution = cur
    
    # Randomized restarts
    for restart in range(20):
        if time.time() >= deadline:
            break
        random.seed(restart)
        added = []
        added_set = set()
        valid, bad_count = check_solution(added)
        if not valid:
            continue
        while bad_count > 0 and time.time() < deadline:
            # Pick among top candidates
            scored = []
            for e in cands:
                if e in added_set:
                    continue
                trial = added + [e]
                valid2, bad2 = check_solution(trial)
                if valid2:
                    scored.append((bad2, e))
            if not scored:
                break
            scored.sort()
            # Pick from top 5
            top = scored[:5]
            _, best_edge = random.choice(top)
            added.append(best_edge)
            added_set.add(best_edge)
            bad_count = check_solution(added)[1]
        
        if bad_count == 0 and len(added) < best_count:
            # Try to reduce
            cur = list(added)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(cur)):
                    if time.time() >= deadline:
                        break
                    trial = cur[:i] + cur[i+1:]
                    valid2, bad2 = check_solution(trial)
                    if valid2 and bad2 == 0:
                        cur = trial
                        improved = True
                        break
            if len(cur) < best_count:
                best_count = len(cur)
                best_solution = cur
    
    if best_solution is None:
        return []
    return [[int(u), int(v)] for (u,v) in best_solution]

def construct():
    result = solve()
    return result
# EVOLVE-BLOCK-END
