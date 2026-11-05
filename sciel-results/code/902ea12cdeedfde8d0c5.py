import time
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    if k != 3:
        return _fallback_algebraic(k, g, deadline)
    return _construct_voltage_search(3, g, deadline)

def _fallback_algebraic(k, g, deadline):
    # Fallback: Return a valid 3-regular graph (may not have high girth, but valid structure)
    # We use a simple random 3-regular graph and verify it.
    # If we can't find one with girth g, we return the best valid graph we can (which might score 0 if girth < g, but avoids crash).
    # However, the task requires girth g. So this is a last resort.
    n = 1028 # Even number
    edges = _random_3reg(n, random.Random(42))
    if edges is None:
        # Try again
        edges = _random_3reg(n, random.Random(43))
    if edges is None:
        # Construct a deterministic 3-regular graph (e.g. cycle with chords)
        # This is hard to do quickly.
        # Return a small valid graph? No, must be large enough?
        # The task says "n must lie between Moore bound and max_vertices".
        # Moore bound for k=3, g=16 is 510.
        # So n >= 510.
        # If we can't find a valid one, we return a graph that is at least simple and 3-regular.
        pass
    return n, edges

def _random_3reg(n, rng):
    if n % 2 != 0:
        n += 1
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i+1]
        if a == b:
            return None
        key = (min(a,b), max(a,b))
        if key in edges:
            return None
        edges.add(key)
    # Verify simple + 3-regular
    deg = [0]*n
    for a,b in edges:
        deg[a]+=1; deg[b]+=1
    if any(d != 3 for d in deg):
        return None
    return list(edges)

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for a,b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj

def _check_girth(adj, n, g):
    # Returns True if girth >= g
    # Use BFS from each vertex
    for start in range(n):
        parent = [-1]*n
        level = [-1]*n
        level[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if level[u] >= (g-1)//2:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if level[v] == -1:
                    level[v] = level[u]+1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and parent[v] != u:
                    cl = level[u] + level[v] + 1
                    if 3 <= cl < g:
                        return False
    return True

def _construct_voltage_search(k, g, deadline):
    rng = random.Random(12345)
    # Base graph: K4
    # Vertices 0,1,2,3
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    # Try different lift sizes m
    # n = 4m
    # Moore bound is 510, so n >= 512 => m >= 128
    # Known best is 936 => m = 234
    # Let's try m values around 128 to 300
    
    m_candidates = list(range(128, 301, 2)) # m must be such that n is even? 4m is always even.
    # Shuffle to try random sizes? Or start from lower?
    # Lower n is better.
    
    for m in m_candidates:
        if time.time() > deadline - 5:
            break
        n = 4 * m
        # Try several random voltage assignments
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Random voltages
            voltages = [rng.randint(0, m-1) for _ in range(6)]
            
            # Construct graph
            edges = set()
            valid = True
            for i, (u, w) in enumerate(base_edges):
                vol = voltages[i]
                for v in range(m):
                    a = v * 4 + u
                    b = ((v + vol) % m) * 4 + w
                    # Check for loops
                    if a == b:
                        valid = False
                        break
                    key = (min(a,b), max(a,b))
                    if key in edges:
                        valid = False
                        break
                    edges.add(key)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check degree
            deg = [0]*n
            for a,b in edges:
                deg[a]+=1
                deg[b]+=1
            if any(d != 3 for d in deg):
                continue
                
            # Check girth
            adj = _build_adj(list(edges), n)
            if _check_girth(adj, n, g):
                return n, list(edges)
                
    # If no voltage graph worked, fall back to random repair (slower)
    return _construct_repair(3, g, deadline)

def _find_short_cycle(adj, n, g):
    # Find a cycle of length < g if exists
    for start in range(n):
        parent = [-1]*n
        level = [-1]*n
        level[start] = 0
        q = deque([start])
        found = None
        while q:
            u = q.popleft()
            if level[u] >= (g-1)//2:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if level[v] == -1:
                    level[v] = level[u]+1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and parent[v] != u:
                    cl = level[u] + level[v] + 1
                    if 3 <= cl < g:
                        # Reconstruct
                        path_u = [u]
                        while path_u[-1] != start:
                            path_u.append(parent[path_u[-1]])
                        path_u.reverse()
                        path_v = [v]
                        while path_v[-1] != start:
                            path_v.append(parent[path_v[-1]])
                        path_v.reverse()
                        cycle = path_u + path_v[1:]
                        found = cycle
                        break
            if found:
                break
        if found:
            return found
    return None

def _try_break_cycle(adj, edges_set, n, cycle, rng, g):
    L = len(cycle)
    if L < 3 or L >= g:
        return False
    # Randomized break
    # Pick random edge in cycle
    for _ in range(10): # Try a few random options
        i = rng.randint(0, L-1)
        a = cycle[i]
        b = cycle[(i+1) % L]
        cand_a = [x for x in adj[a] if x != b]
        if not cand_a: continue
        x = rng.choice(cand_a)
        cand_b = [y for y in adj[b] if y != a and y != x]
        if not cand_b: continue
        y = rng.choice(cand_b)
        if (min(x,y), max(x,y)) not in edges_set:
            continue
        # Proposed: remove (a,b) and (x,y), add (a,y) and (x,b)
        if a == y or x == b:
            continue
        if (min(a,y), max(a,y)) in edges_set:
            continue
        if (min(x,b), max(x,b)) in edges_set:
            continue
        edges_set.discard((min(a,b), max(a,b)))
        edges_set.discard((min(x,y), max(x,y)))
        edges_set.add((min(a,y), max(a,y)))
        edges_set.add((min(x,b), max(x,b)))
        adj[a].remove(b); adj[b].remove(a)
        adj[x].remove(y); adj[y].remove(x)
        adj[a].append(y); adj[y].append(a)
        adj[x].append(b); adj[b].append(x)
        return True
    return False

def _construct_repair(k, g, deadline):
    rng = random.Random(42)
    for n0 in [512, 514, 516, 520, 524, 528, 532, 536, 540, 548, 556, 564, 572, 580, 588, 600]:
        if time.time() > deadline - 10:
            break
        edges_list = _random_3reg(n0, rng)
        if edges_list is None:
            continue
        edges_set = set(edges_list)
        adj = _build_adj(edges_list, n0)
        for it in range(50000):
            if time.time() > deadline - 15:
                break
            cycle = _find_short_cycle(adj, n0, g)
            if cycle is None:
                return n0, list(edges_set)
            if not _try_break_cycle(adj, edges_set, n0, cycle, rng, g):
                break
    # Last resort
    return _fallback_algebraic(k, g, deadline)
