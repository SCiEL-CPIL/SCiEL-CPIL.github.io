"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque
import numpy as np

def _verify_girth_fast(adj, n, g):
    """Fast girth check using BFS from each vertex, stopping at depth g/2.
    Returns True if girth >= g."""
    # For 3-regular graphs, we can use non-backtracking walks.
    # But for g=16, depth is 8.
    # Number of walks is 3 * 2^7 = 384 per vertex.
    # Total 384 * n. For n=1000, 384k ops. Very fast.
    
    for start in range(n):
        # State: (current, prev)
        # We want to detect if we return to start with length < g.
        # Actually, we just need to check if any non-backtracking walk of length < g closes.
        # We can do a BFS on the state space (v, prev).
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                # If w is start and d+1 >= 3, we found a cycle
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _generate_random_3reg(n, rng):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n % 2 != 0:
        return None
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(0, n * 3, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    # Check connectivity
    visited = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
        
    return adj, edges

def _find_short_cycles(adj, n, max_len):
    """Find all cycles of length < max_len. Returns list of cycles (list of vertices)."""
    cycles = []
    for start in range(n):
        # BFS from start
        # We want to find paths from start to neighbors of start with length < max_len - 1
        # Actually, we can just run the girth check and record the cycle.
        # But for repair, we just need one short cycle to break.
        # Let's find ANY short cycle.
        dist = {}
        prev = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        prev[(start, -1)] = None
        
        found = False
        while dq and not found:
            v, p = dq.popleft()
            d = dist[(v, p)]
            if d >= max_len - 1:
                continue
            for w in adj[v]:
                if w == p:
                    continue
                if w == start and d + 1 >= 3:
                    # Found cycle
                    # Reconstruct path
                    path = [w]
                    cur = (v, p)
                    while cur is not None:
                        path.append(cur[0])
                        cur = prev[cur]
                    # path is [start, v, ..., start] ?
                    # Actually, the cycle is the path from start to w (which is start)
                    # The vertices in the cycle are path[1:] (excluding the final start duplicate?)
                    # Let's just return the vertices involved.
                    cycles.append(path[1:])
                    found = True
                    break
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    prev[state] = (v, p)
                    dq.append(state)
    return cycles

def _attempt_repair(adj, n, g, rng, max_iters=1000):
    """Try to repair the graph to have girth >= g using 2-switches."""
    # Copy adj to modify
    adj = [set(s) for s in adj]
    
    for _ in range(max_iters):
        # Check girth
        if _verify_girth_fast(adj, n, g):
            return adj
        
        # Find a short cycle
        cycles = _find_short_cycles(adj, n, g)
        if not cycles:
            # Girth check said fail, but no cycle found? Should not happen.
            # Maybe the girth check found a cycle that _find_short_cycles missed?
            # _find_short_cycles finds cycles of length < g.
            # _verify_girth_fast returns False if any cycle < g.
            # So if verify fails, there must be a cycle.
            return None
        
        # Pick a random short cycle
        cycle = rng.choice(cycles)
        # cycle is a list of vertices [v0, v1, ..., vk] where v0 is start?
        # My _find_short_cycles returns path[1:] which is the vertices on the path from start to w.
        # If w==start, the path is start -> ... -> start.
        # The cycle vertices are the unique vertices in the path.
        
        # To break the cycle, we can pick two edges in the cycle and swap them with other edges.
        # A 2-switch: remove (u,v) and (x,y), add (u,x) and (v,y).
        # We need (u,x) and (v,y) to not exist and not be loops.
        
        # Let's pick two edges from the cycle.
        # The cycle edges are (cycle[i], cycle[i+1]) and (cycle[-1], cycle[0])?
        # The list `cycle` from _find_short_cycles is the path.
        # Let's just pick two random edges incident to vertices in the cycle, 
        # ensuring they are part of the cycle or connect to it.
        
        # Simpler: Pick two random edges in the whole graph, 
        # but bias towards edges in the cycle.
        # Let's pick two edges from the cycle.
        # The cycle length is L.
        # Edges are (cycle[i], cycle[(i+1)%L])? 
        # Note: `cycle` from my function is the path vertices. 
        # If the path is v0(start), v1, ..., vk(w=start), then edges are (v0,v1), (v1,v2)...(v_{k-1},vk).
        # Since vk == v0, the edge (v_{k-1}, vk) is (v_{k-1}, v0).
        # So the cycle edges are (cycle[i], cycle[i+1]) for i in 0..L-2.
        
        L = len(cycle)
        if L < 3:
            continue
            
        # Pick two distinct edges
        e1_idx = rng.randint(0, L-1)
        e2_idx = rng.randint(0, L-1)
        if e1_idx == e2_idx:
            continue
            
        u1, v1 = cycle[e1_idx], cycle[e1_idx+1]
        u2, v2 = cycle[e2_idx], cycle[e2_idx+1]
        
        # Try to swap
        # Remove (u1, v1) and (u2, v2)
        # Add (u1, u2) and (v1, v2) ? Or (u1, v2) and (v1, u2)?
        # We need to ensure no multi-edges and no loops.
        
        for swap_type in range(2):
            if swap_type == 0:
                new_u1, new_v1 = u1, u2
                new_u2, new_v2 = v1, v2
            else:
                new_u1, new_v1 = u1, v2
                new_u2, new_v2 = v1, u2
                
            if new_u1 == new_v1 or new_u2 == new_v2:
                continue
            if new_v1 in adj[new_u1] or new_v2 in adj[new_u2]:
                continue
                
            # Apply swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[new_u1].add(new_v1)
            adj[new_v1].add(new_u1)
            adj[new_u2].add(new_v2)
            adj[new_v2].add(new_u2)
            
            # Check if this improved things (e.g. broke the cycle)
            # We just continue the loop; the next iteration will check girth.
            # To avoid oscillation, we could check if the specific cycle is gone.
            break

    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        # Fallback to LUW for other k values
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
            add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # For k=3: Use Random 3-regular graphs with Repair
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
        
    # Try n from Moore bound upwards
    # Step size 2 to keep n even
    n = moore
    if n % 2 != 0:
        n += 1
        
    rng = _random.Random(42)
    
    while n < 20000 and _time.time() < deadline - 5:
        # Try multiple random graphs for this n
        max_attempts = 50
        for attempt in range(max_attempts):
            if _time.time() > deadline - 3:
                break
                
            result = _generate_random_3reg(n, rng)
            if result is None:
                continue
                
            adj, edges = result
            
            # Try to repair
            repaired_adj = _attempt_repair(adj, n, g, rng, max_iters=200)
            
            if repaired_adj is not None:
                # Verify again to be sure
                if _verify_girth_fast(repaired_adj, n, g):
                    # Reconstruct edges from repaired_adj
                    new_edges = []
                    for u in range(n):
                        for v in repaired_adj[u]:
                            if u < v:
                                new_edges.append((u, v))
                    return n, new_edges
        
        n += 2
        
    # Fallback: Return the first valid random graph found without repair?
    # Or just the LUW construction as before.
    # Let's try one more random graph at a larger n just in case.
    n = max(n, 1000)
    while n < 10000 and _time.time() < deadline - 5:
        result = _generate_random_3reg(n, rng)
        if result is not None:
            adj, edges = result
            if _verify_girth_fast(adj, n, g):
                return n, edges
        n += 2
        
    # Final Fallback: LUW construction
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
