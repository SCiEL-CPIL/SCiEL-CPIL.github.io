"""Construct a k-regular graph of girth at least g on as few vertices as possible."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import deque

def _moore_bound(k, g):
    """Lower bound on number of vertices for a k-regular graph of girth g."""
    if g % 2 == 0:
        r = g // 2 - 1
        return 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        return 2 * sum((k - 1) ** i for i in range(r))

def _bfs_girth(adj):
    """Compute the girth (shortest cycle) of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] * 2 >= best:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best:
                        best = cycle_len
    return best if best != float('inf') else 2 * n + 1

def _verify_graph(n, edges, k, g):
    """Verify: simple, k-regular, girth >= g."""
    edge_set = set()
    deg = [0] * n
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        deg[i] += 1
        deg[j] += 1
    for d in deg:
        if d != k:
            return False
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    girth = _bfs_girth(adj)
    return girth >= g

def _lazebnik_ustimenko_woldar(k, g):
    """Construct LZW graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    if d < 3:
        d = 3

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for idx, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def _girth_ok_cayley(k, g, S, q):
    """Check if Cayley graph on Z_q with connection set S has girth >= g.
    S must be symmetric and not contain 0.
    """
    S_list = list(S)
    nS = len(S_list)
    S_neg_map = {s: (-s) % q for s in S_list}
    
    # DP: set of (last_gen_index, current_sum)
    # Start with length 0? No, we look for closed walks of length L >= 2.
    # A closed walk of length L is s1...sL such that sum=0 and s_i != -s_{i+1} (cyclic).
    # We check lengths 2 to g-1.
    
    # For length 2: s1, s2. s1+s2=0 => s2=-s1. s2 != -s1 is violated.
    # So no closed non-backtracking walks of length 2.
    
    # For length 3: s1, s2, s3. s1+s2+s3=0. s2 != -s1, s3 != -s2, s1 != -s3.
    
    depth_states = set()
    # Initialize for length 1?
    # Actually, let's just iterate lengths.
    # State: (last_s, sum_so_far)
    # We want to know if we can reach sum=0 with last_s != -first_s?
    # It's easier to track (first_s, last_s, sum).
    # But that's O(k^2 * q) states.
    
    # Let's use the method from Inspiration 2:
    # depth0 = {(0, 0)} is not right.
    
    # Let's do:
    # For each length L from 2 to g-1:
    #   Check if there exists s1...sL in S such that sum=0 and s_i != -s_{i+1} and s_L != -s_1.
    
    # Optimization: For L=3, 4, etc.
    # Since q is small (~500), we can just enumerate.
    # Number of sequences is k^L. For L=8, 5^8 = 400k.
    # We can prune.
    
    # Let's use a set of reachable (last_s, sum) pairs for each length,
    # but we also need to remember the first_s to check the closure condition s_L != -s_1.
    # So state is (first_s, last_s, sum).
    # Size: k * k * q = 25 * 500 = 12500 states.
    # This is very small.
    
    for L in range(3, g): # Length 2 is impossible for non-backtracking
        if L == 3:
            # s1, s2, s3
            # s1+s2+s3=0
            # s2 != -s1, s3 != -s2, s1 != -s3
            found = False
            for s1 in S_list:
                for s2 in S_list:
                    if s2 == (-s1) % q: continue
                    s3 = (-s1 - s2) % q
                    if s3 in S_neg_map.values() or s3 in S_list: # s3 must be in S
                        # Check s3 in S
                        if s3 in S_list:
                            if s3 != (-s2) % q and s1 != (-s3) % q:
                                return False
            # If no cycle found for L=3, continue to L=4
        else:
            # DP for length L
            # States from L-1: (first_s, last_s, sum)
            # We need to store states from previous step.
            pass
    
    # The above is getting complex. Let's use the simpler BFS on the non-backtracking graph?
    # Or just use the DP from Inspiration 2 which was:
    # depth0 = {(0, 0)}
    # for length in range(2, target+1):
    #    ...
    # This checked if sum=0 was reachable. It didn't check the closure condition s_L != -s_1.
    # But if sum=0 is reachable by a non-backtracking walk, does it imply a cycle?
    # Yes, if the walk is s1...sL, sum=0.
    # If s_L != -s_1, it's a cycle.
    # If s_L == -s_1, it's a backtrack at the end?
    # No, non-backtracking means s_i != -s_{i+1}.
    # If s_L = -s_1, then the walk s1...sL is a closed walk that backtracks on the first edge.
    # This is not a simple cycle.
    # So we MUST check s_L != -s_1.
    
    # Let's implement the (first_s, last_s, sum) DP.
    
    # For L=3, we did it manually.
    # For L>=4:
    prev_states = set()
    for s1 in S_list:
        for s2 in S_list:
            if s2 == (-s1) % q: continue
            prev_states.add((s1, s2, (s1+s2)%q))
            
    for L in range(4, g):
        curr_states = set()
        for first_s, last_s, s_sum in prev_states:
            for s_next in S_list:
                if s_next == (-last_s) % q: continue
                new_sum = (s_sum + s_next) % q
                # Check closure: new_sum == 0 and s_next != -first_s
                if new_sum == 0 and s_next != (-first_s) % q:
                    return False
                curr_states.add((first_s, s_next, new_sum))
        prev_states = curr_states
        if not prev_states:
            break
            
    return True

def _construct_cayley_cage(k, g, max_q=2000):
    """Search for a Cayley graph on Z_q with girth >= g."""
    if k % 2 == 1:
        # q must be even
        start_q = _moore_bound(k, g)
        if start_q % 2 != 0:
            start_q += 1
    else:
        start_q = _moore_bound(k, g)
        
    for q in range(start_q, max_q, 2 if k % 2 == 1 else 1):
        # Try to find S
        # For odd k, S = {a, b, q/2, -a, -b}
        # For even k, S = {a, b, c, -a, -b, -c}
        
        attempts = 200
        for _ in range(attempts):
            if k % 2 == 1:
                # Pick a, b randomly
                a = random.randint(1, q-1)
                b = random.randint(1, q-1)
                if a == b or a == q-a or b == q-b:
                    continue
                if a == q-b or b == q-a:
                    continue
                S = {a, b, q//2, (-a)%q, (-b)%q}
                if len(S) != k or 0 in S:
                    continue
            else:
                a = random.randint(1, q-1)
                b = random.randint(1, q-1)
                c = random.randint(1, q-1)
                S = {a, b, c, (-a)%q, (-b)%q, (-c)%q}
                if len(S) != k or 0 in S:
                    continue
                
            if _girth_ok_cayley(k, g, S, q):
                # Construct edges
                edges = []
                seen = set()
                for u in range(q):
                    for s in S:
                        v = (u + s) % q
                        if u < v:
                            edge = (u, v)
                            if edge not in seen:
                                seen.add(edge)
                                edges.append(edge)
                return q, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Try Cayley graph search for small k, g
    if k <= 10 and g <= 10:
        try:
            result = _construct_cayley_cage(k, g)
            if result:
                n, edges = result
                if _verify_graph(n, edges, k, g):
                    return n, edges
        except Exception:
            pass

    if time.time() > deadline:
        return _lazebnik_ustimenko_woldar(k, g)

    # Fallback to LUW
    return _lazebnik_ustimenko_woldar(k, g)
# EVOLVE-BLOCK-END
