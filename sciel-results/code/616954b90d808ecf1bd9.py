"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    deadline = time.time() + 140
    
    def compute_girth(adj, n, target):
        """Compute girth using BFS. Returns early if girth <= target."""
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth <= target:
                                return min_girth
        return min_girth
    
    def voltage_lift_petersen(group_order, voltages):
        """Petersen voltage lift. voltages: list of 15 (one per edge) in Z_m."""
        n = 10 * group_order
        adj = [[] for _ in range(n)]
        petersen_edges = [
            (0,1),(1,2),(2,3),(3,4),(4,0),
            (5,7),(7,9),(9,6),(6,8),(8,5),
            (0,5),(1,7),(2,9),(3,6),(4,8)
        ]
        for i, (u,v) in enumerate(petersen_edges):
            vol = voltages[i] % group_order
            for coset in range(group_order):
                u_lift = u * group_order + coset
                v_lift = v * group_order + (coset + vol) % group_order
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
        return adj
    
    # Petersen has 15 edges
    n_edges = 15
    
    # Try voltage lifts over Petersen
    for group_order in range(90, 130, 2):
        if time.time() > deadline - 15:
            break
        n = 10 * group_order
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            voltages = [random.randint(0, group_order - 1) for _ in range(n_edges)]
            adj = voltage_lift_petersen(group_order, voltages)
            girth = compute_girth(adj, n, g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: LUW construction
    q = 2
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    add = lambda x,y: (x+y)%q
    sub = lambda x,y: (x-y)%q
    mul = lambda x,y: x*y%q
    rule = {2:(1,1),3:(2,1),4:(1,2)}
    for c in range(5, d+1):
        r = (c-5)%4
        c0 = c - (c-5)%4
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
            if vertex[0]==0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
