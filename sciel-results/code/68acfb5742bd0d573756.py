"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

N = 16
DEADLINE = 110.0  # seconds; leave margin before 120s limit

def _time_left():
    return DEADLINE - time.time()

def _check_not_3colorable(adj, time_limit=1.5):
    """Backtracking 3-coloring with pruning. Returns True if NOT 3-colorable."""
    # Order vertices by degree descending
    order = sorted(range(N), key=lambda v: -len(adj[v]))
    pos = {v: i for i, v in enumerate(order)}
    color = [-1] * N  # -1 uncolored, 0/1/2 colored
    node_count = [0]
    start = time.time()
    
    def _feasible(v, c):
        for u in adj[v]:
            if color[u] == c:
                return False
        return True
    
    def _dfs(idx):
        node_count[0] += 1
        if node_count[0] % 20000 == 0:
            if time.time() - start > time_limit:
                return 2  # timeout - inconclusive
        if idx == N:
            return 1  # colored successfully -> IS 3-colorable
        v = order[idx]
        # MRV: check which colors are available
        avail = []
        for c in range(3):
            if _feasible(v, c):
                avail.append(c)
        if not avail:
            return 0  # dead end
        # Try colors in order of most constrained (fewest options for neighbors)
        for c in avail:
            color[v] = c
            # Forward check
            conflict = False
            for u in adj[v]:
                if color[u] == -1 and pos[u] > idx:
                    # Check if u still has at least one available color
                    u_avail = 0
                    for c2 in range(3):
                        ok = True
                        for w in adj[u]:
                            if color[w] == c2:
                                ok = False
                                break
                        if ok:
                            u_avail += 1
                    if u_avail == 0:
                        conflict = True
                        break
            if not conflict:
                r = _dfs(idx + 1)
                if r != 0:
                    color[v] = -1
                    return r
            color[v] = -1
        return 0
    
    r = _dfs(0)
    if r == 1:
        return False  # IS 3-colorable
    if r == 2:
        return None   # inconclusive (timeout)
    return True       # NOT 3-colorable

def _is_triangle_free(adj):
    for u in range(N):
        for v in adj[u]:
            if u < v:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True

def _edge_count(adj):
    return sum(len(adj[i]) for i in range(N)) // 2

def _build_grotzsch_blowup():
    """Build the 47-edge Grötzsch blow-up."""
    # Grötzsch graph structure (Mycielski of C5):
    # Vertices 0-4: cycle
    # Vertices 5-9: shadows (5i adjacent to 4i and 4(i+1))
    # Vertex 10: connected to all shadows 5-9
    #
    # Blow-up: 5->2 copies, 10->5 copies, others->1 copy
    # Total: 1+1+1+1+1+2+1+1+1+1+5 = 16
    
    # Map slot -> list of vertex indices
    slot_map = {
        0: [0], 1: [1], 2: [2], 3: [3], 4: [4],
        5: [5, 6], 6: [7], 7: [8], 8: [9], 9: [10],
        10: [11, 12, 13, 14, 15]
    }
    
    # Grötzsch edges between slots (u<v in slot numbering)
    slot_edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),  # cycle
        (0,5),(1,5),(1,6),(2,6),(2,7),(3,7),(3,8),(4,8),(4,9),(0,9),  # shadow connections
        (5,10),(6,10),(7,10),(8,10),(9,10)  # root connections
    ]
    
    adj = [set() for _ in range(N)]
    for su, sv in slot_edges:
        for u in slot_map[su]:
            for v in slot_map[sv]:
                adj[u].add(v)
                adj[v].add(u)
    
    return adj

def _greedy_add_edges(adj, time_budget=30.0):
    """Try to add edges greedily while maintaining triangle-free + not 3-colorable."""
    adj = [set(s) for s in adj]
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        # Try all non-edges
        candidates = []
        for u in range(N):
            for v in range(u+1, N):
                if v not in adj[u]:
                    candidates.append((u, v))
        random.shuffle(candidates)
        for u, v in candidates:
            if time.time() - start > time_budget:
                break
            # Check triangle-free
            common = adj[u] & adj[v]
            if common:
                continue
            # Try adding edge
            adj[u].add(v)
            adj[v].add(u)
            # Check not 3-colorable
            r = _check_not_3colorable(adj, time_limit=1.0)
            if r is True:
                improved = True
            else:
                adj[u].remove(v)
                adj[v].remove(u)
    return adj

def _local_search_swap(adj, time_budget=60.0):
    """Try swap moves: remove one edge, add another. Also try single edge additions."""
    adj = [set(s) for s in adj]
    start = time.time()
    
    while time.time() - start < time_budget:
        best_adj = [set(s) for s in adj]
        best_count = _edge_count(adj)
        
        # Build list of all possible single-edge additions that are triangle-safe
        add_candidates = []
        for u in range(N):
            for v in range(u+1, N):
                if v not in adj[u] and not (adj[u] & adj[v]):
                    add_candidates.append((u, v))
        
        # Try each addition
        for u, v in add_candidates:
            if time.time() - start > time_budget:
                break
            adj[u].add(v)
            adj[v].add(u)
            r = _check_not_3colorable(adj, time_limit=1.0)
            if r is True and _edge_count(adj) > best_count:
                best_count = _edge_count(adj)
                best_adj = [set(s) for s in adj]
            else:
                adj[u].remove(v)
                adj[v].remove(u)
        
        # Try swap moves: for each edge (a,b) in adj, try removing it and adding a non-edge
        if time.time() - start > time_budget:
            break
        
        # Get current edges
        cur_edges = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    cur_edges.append((u, v))
        
        random.shuffle(cur_edges)
        found_swap = False
        for a, b in cur_edges:
            if time.time() - start > time_budget:
                break
            if found_swap:
                break
            # Remove edge (a,b)
            adj[a].remove(b)
            adj[b].remove(a)
            
            # Try adding a non-edge
            for u in range(N):
                if found_swap:
                    break
                for v in range(u+1, N):
                    if time.time() - start > time_budget:
                        break
                    if v in adj[u]:
                        continue
                    if u == a and v == b:
                        continue
                    if adj[u] & adj[v]:
                        continue
                    adj[u].add(v)
                    adj[v].add(u)
                    r = _check_not_3colorable(adj, time_limit=1.0)
                    if r is True and _edge_count(adj) > best_count:
                        best_count = _edge_count(adj)
                        best_adj = [set(s) for s in adj]
                        found_swap = True
                    else:
                        adj[u].remove(v)
                        adj[v].remove(u)
                    if found_swap:
                        break
            
            # Restore original edge
            adj[a].add(b)
            adj[b].add(a)
        
        if not found_swap:
            # No improvement found; break
            break
        
        adj = [set(s) for s in best_adj]
    
    return adj

def construct():
    random.seed(42)
    start_time = time.time()
    
    # Start with the 47-edge blow-up
    adj = _build_grotzsch_blowup()
    best_adj = [set(s) for s in adj]
    best_count = _edge_count(adj)
    
    # Verify our starting point
    assert _is_triangle_free(adj), "Start is not triangle-free"
    r = _check_not_3colorable(adj, time_limit=5.0)
    if r is not True:
        # Fallback to original Grötzsch
        cycle = [(i, (i+1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5+w], [w, 5+u]]
        edges += [[5+u, 10] for u in range(5)]
        return edges
    
    # Phase 1: Greedy edge addition
    remaining = DEADLINE - time.time() - 70.0  # leave 70s for local search
    if remaining > 5:
        adj = _greedy_add_edges(adj, time_budget=remaining)
        if _edge_count(adj) > best_count:
            best_count = _edge_count(adj)
            best_adj = [set(s) for s in adj]
    
    # Phase 2: Local search (swaps + additions)
    remaining = DEADLINE - time.time() - 10.0  # leave 10s margin
    if remaining > 5:
        adj = _local_search_swap(adj, time_budget=remaining)
        if _edge_count(adj) > best_count:
            best_count = _edge_count(adj)
            best_adj = [set(s) for s in adj]
    
    # Final verification
    if not _is_triangle_free(best_adj):
        # Fallback
        cycle = [(i, (i+1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5+w], [w, 5+u]]
            edges += [[5+u, 10] for u in range(5)]
        return edges
    
    r = _check_not_3colorable(best_adj, time_limit=5.0)
    if r is not True:
        # Fallback
        cycle = [(i, (i+1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5+w], [w, 5+u]]
            edges += [[5+u, 10] for u in range(5)]
        return edges
    
    # Convert to edge list
    edge_list = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edge_list.append([u, v])
    
    return edge_list
# EVOLVE-BLOCK-END
