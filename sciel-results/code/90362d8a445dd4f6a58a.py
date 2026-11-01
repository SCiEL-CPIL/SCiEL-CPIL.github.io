"""Construct a small 3-regular graph of girth g via configuration model + simulated annealing."""
import time
import random
import math
from collections import deque

_DEADLINE = 150.0  # seconds total budget
_SAFETY = 5.0      # return at least this many seconds before kill

def _moore_bound(k, g):
    """Moore bound for k-regular graph of girth g."""
    if g % 2 == 0:
        # even girth: n >= k * (k-1)^(g/2 - 1) / (k-1) ... actually:
        # n >= k * sum_{i=0}^{g/2-1} (k-1)^i = k * ((k-1)^(g/2) - 1) / (k-2) for k>2
        # Wait, standard Moore bound: for even girth g, n >= k*(k-1)^{g/2-1} ... no.
        # Correct: n >= k * ( (k-1)^(g/2) - 1 ) / (k-2) for k > 2, g even
        # For k=3: n >= 3 * ((2)^(g/2) - 1) / 1 = 3*(2^(g/2)-1)
        # For g=16: 3*(256-1) = 765. But the task says best lower bound is 512.
        # Actually the Moore bound for (3,16): 3*(2^8 - 1) = 765? That seems high.
        # Let me re-check: Moore bound for k-regular, girth g:
        # if g even: n >= k * (k-1)^{g/2 - 1} ... no that's not right either.
        # The correct Moore bound: for even girth g = 2r: n >= k * ((k-1)^r - 1)/(k-2)
        # For k=3, g=16, r=8: n >= 3*(2^8-1)/1 = 3*255 = 765.
        # But the task says lower bound is 512. Maybe they use a different bound.
        # I'll just use a reasonable lower bound for our search range.
        r = g // 2
        if k > 2:
            return int(k * ((k - 1) ** r - 1) / (k - 2))
        else:
            return k * (2 ** r - 1)
    else:
        # odd girth g = 2r+1: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i + k*(k-1)^r
        # = 1 + k * ((k-1)^r - 1)/(k-2) + k*(k-1)^r for k>2
        r = (g - 1) // 2
        if k > 2:
            return int(1 + k * ((k - 1) ** r - 1) / (k - 2) + k * (k - 1) ** r)
        else:
            return 1 + k * (2 ** r - 1) + k * 2 ** r


def _gen_random_3reg(n, rng):
    """Generate a random 3-regular graph on n vertices using configuration model.
    Returns adjacency list (list of sets) or None if not simple."""
    if n < 4 or n % 2 != 0:
        return None
    # Each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    
    for _ in range(100):  # try up to 100 times to get a simple graph
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            # Convert to sorted lists for faster access
            return [sorted(s) for s in adj]
    return None


def _girth_bfs(adj, max_girth):
    """Compute girth via non-backtracking BFS from each vertex.
    Returns the girth (minimum cycle length), or infinity if none found below max_girth."""
    n = len(adj)
    best = max_girth + 1  # we only care if girth < max_girth
    
    for start in range(n):
        # BFS with non-backtracking: state is (vertex, prev_vertex)
        # dist[prev][v] = distance
        from collections import defaultdict
        # Use a dict: (prev, v) -> dist
        dist = {}
        queue = deque()
        for nb in adj[start]:
            dist[(start, nb)] = 2
            queue.append((start, nb))
        
        while queue:
            prev, v = queue.popleft()
            d = dist[(prev, v)]
            if d >= best:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if (v, nb) in dist:
                    # Found a cycle of length d + 1
                    if d + 1 < best:
                        best = d + 1
                    # Don't need to enqueue again
                else:
                    dist[(v, nb)] = d + 1
                    queue.append((v, nb))
        if best <= max_girth:
            return best
    return best


def _girth_fast(adj, max_girth):
    """Faster girth check: BFS from each vertex, tracking shortest cycle.
    Uses the fact that for girth g, we only need to explore up to depth g/2 from each vertex."""
    n = len(adj)
    best = max_girth + 1
    half = max_girth // 2
    
    for start in range(n):
        # BFS from start, tracking parent to detect cycles
        # dist[v] = shortest distance from start
        # parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            v = queue.popleft()
            if dist[v] >= half:
                continue
            for nb in adj[v]:
                if dist[nb] == -1:
                    dist[nb] = dist[v] + 1
                    parent[nb] = v
                    queue.append(nb)
                elif nb != parent[v] and parent[nb] != v:
                    # Found a cycle: dist[v] + dist[nb] + 1
                    cyc = dist[v] + dist[nb] + 1
                    if cyc < best:
                        best = cyc
        if best <= max_girth:
            return best
    return best


def _count_short_cycles(adj, max_len):
    """Count the number of cycles of length < max_len. Used as energy."""
    n = len(adj)
    count = 0
    # For small cycles (3,4,5,...), we can count them directly
    # But for general max_len up to 15, this is expensive.
    # Instead, use girth-based approach: just check if girth < max_len
    g = _girth_fast(adj, max_len)
    if g < max_len:
        # Count cycles of length g (approximate by just returning a penalty)
        return max_len - g
    return 0


def _swap_edge(adj, e1, e2, rng):
    """Perform an edge swap: remove edges (u,v) and (x,y), add (u,x) and (v,y) or (u,y) and (v,x).
    e1 = (u,v), e2 = (x,y). Returns True if swap was valid (no multiedges, no loops)."""
    u, v = e1
    x, y = e2
    # Try both possible reconnections
    for perm in [(u, x, v, y), (u, y, v, x)]:
        a, b, c, d = perm
        if a == b or c == d:
            continue
        if b in adj[a] and d in adj[c]:
            # Would create multi-edge
            continue
        if d in adj[a] or b in adj[c]:
            continue
        # Perform swap
        adj[a].remove(b)
        adj[a].append(d)
        adj[c].remove(d)
        adj[c].append(b)
        adj[b].remove(a)
        adj[b].append(c)
        adj[d].remove(c)
        adj[d].append(a)
        # Keep sorted
        adj[a].sort()
        adj[b].sort()
        adj[c].sort()
        adj[d].sort()
        return True
    return False


def _undo_swap(adj, e1, e2, perm):
    """Undo a swap."""
    a, b, c, d = perm
    adj[a].remove(d)
    adj[a].append(b)
    adj[c].remove(b)
    adj[c].append(d)
    adj[b].remove(c)
    adj[b].append(a)
    adj[d].remove(a)
    adj[d].append(c)
    adj[a].sort()
    adj[b].sort()
    adj[c].sort()
    adj[d].sort()


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    t0 = time.time()
    deadline = t0 + _DEADLINE - _SAFETY
    
    if k != 3:
        # Fallback: use the original LZW construction for other k
        return _construct_lzw(k, g)
    
    # For k=3, g=16: Moore bound is 765, but known best is 936.
    # Search range: from Moore bound to ~1000
    # Actually the task says best lower bound is 512, so let's start there.
    # But we need n to be even for 3-regular graphs.
    
    rng = random.Random(42)
    
    # We'll try different n values, starting from smaller ones
    # For 3-regular, n must be even.
    # Start from a reasonable lower bound. The Moore bound for (3,16) is 765.
    # But let's try from 512 (the stated lower bound) upward.
    # Actually, to save time, let's focus on a few promising n values.
    
    # Strategy: try n values, for each do simulated annealing
    # to try to achieve girth >= g.
    
    best_result = None  # (n, edges)
    
    # Candidate n values (even, >= Moore bound-ish)
    # Let's try a range of n values
    n_start = 512
    n_end = 1000
    # Step by 2 (must be even)
    
    # To save time, try fewer n values with more SA iterations each
    # Let's try n = 512, 514, ..., but with time budgeting
    
    # Actually, let's be smarter: try n values in increasing order,
    # and for each, spend a fixed time budget on SA.
    
    time_per_n = 10.0  # seconds per n value
    max_attempts_per_n = 20  # random restarts per n
    
    n = n_start
    while n <= n_end and time.time() < deadline:
        if n % 2 != 0:
            n += 1
            continue
        
        n_t0 = time.time()
        n_deadline = min(n_t0 + time_per_n, deadline)
        
        for attempt in range(max_attempts_per_n):
            if time.time() >= n_deadline:
                break
            
            adj = _gen_random_3reg(n, rng)
            if adj is None:
                continue
            
            # Check initial girth
            girth = _girth_fast(adj, g)
            if girth >= g:
                # Found a valid graph!
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (n, edges)
                return best_result
            
            # Simulated annealing
            # Energy = number of short cycles (approximate by girth deficit)
            # We'll use a simpler approach: try random edge swaps and accept
            # if girth doesn't decrease
            
            temp = 1.0
            cooling = 0.999
            current_energy = g  # we want girth >= g, so energy = max(0, g - girth)
            # Actually let's define energy as (g - girth) if girth < g, else 0
            current_energy = max(0, g - girth)
            
            sa_deadline = min(n_deadline, time.time() + (time_per_n / max_attempts_per_n) * 0.8)
            
            while time.time() < sa_deadline:
                # Pick two random edges
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
                
                if len(all_edges) < 2:
                    break
                
                e1 = rng.choice(all_edges)
                e2 = rng.choice(all_edges)
                if e1 == e2:
                    continue
                
                # Try the swap
                u, v = e1
                x, y = e2
                # Check they're disjoint
                if len({u, v, x, y}) < 4:
                    continue
                
                # Try both reconnections
                swapped = False
                for perm in [(u, x, v, y), (u, y, v, x)]:
                    a, b, c, d = perm
                    if a == b or c == d:
                        continue
                    if d in adj[a] or b in adj[c]:
                        continue
                    # Perform swap
                    adj[a].remove(b)
                    adj[a].append(d)
                    adj[c].remove(d)
                    adj[c].append(b)
                    adj[b].remove(a)
                    adj[b].append(c)
                    adj[d].remove(c)
                    adj[d].append(a)
                    adj[a].sort()
                    adj[b].sort()
                    adj[c].sort()
                    adj[d].sort()
                    swapped = True
                    break
                
                if not swapped:
                    continue
                
                # Compute new girth
                new_girth = _girth_fast(adj, g)
                new_energy = max(0, g - new_girth)
                
                # Accept if better or with probability exp(-(new_energy - current_energy)/temp)
                if new_energy <= current_energy:
                    current_energy = new_energy
                else:
                    if rng.random() < math.exp(-(new_energy - current_energy) / temp):
                        current_energy = new_energy
                    else:
                        # Undo swap
                        a, b, c, d = perm
                        adj[a].remove(d)
                        adj[a].append(b)
                        adj[c].remove(b)
                        adj[c].append(d)
                        adj[b].remove(c)
                        adj[b].append(a)
                        adj[d].remove(a)
                        adj[d].append(c)
                        adj[a].sort()
                        adj[b].sort()
                        adj[c].sort()
                        adj[d].sort()
                
                temp *= cooling
                if temp < 0.01:
                    break
            
            # After SA, check if we achieved girth >= g
            final_girth = _girth_fast(adj, g)
            if final_girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (n, edges)
                return best_result
        
        n += 2
    
    # If we didn't find a valid graph, return the best we can
    # Fallback: use the LZW construction
    if best_result is not None:
        return best_result
    
    return _construct_lzw(k, g)


def _construct_lzw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
