"""Packing edge-disjoint Hamiltonian cycles on the fixed 21-vertex host graph."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
BASE_ADJ = [set() for _ in range(N)]
EDGE_SET = set()
for a, b in P['edges']:
    BASE_ADJ[a].add(b)
    BASE_ADJ[b].add(a)
    EDGE_SET.add((a, b) if a < b else (b, a))


def _check_connected(adj, n):
    if n == 0:
        return True
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == n


def _find_hamiltonian(adj, deadline, rng, forbidden_edges=set(), max_steps=2000000):
    """Find one Hamiltonian cycle using only edges in adj, avoiding forbidden_edges.
    Returns list of vertices (length N, distinct) or None."""
    n = N
    # quick degree feasibility
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    # connectivity check
    if not _check_connected(adj, n):
        return None

    best_cycle = None
    steps = 0
    start = 0
    path = [start]
    used = [False] * n
    used[start] = True

    def backtrack():
        nonlocal steps, best_cycle
        if best_cycle is not None:
            return True
        if time.time() >= deadline:
            return False
        if len(path) == n:
            cur = path[-1]
            if start in adj[cur]:
                e = (start, cur) if start < cur else (cur, start)
                if e not in forbidden_edges:
                    best_cycle = list(path)
                    return True
            return False
        cur = path[-1]
        cand = [x for x in adj[cur] if not used[x]]
        if not cand:
            return False

        # Warnsdorff: prefer candidates with fewer forward options
        # Also filter out forbidden edges in the candidate selection if possible?
        # The check is done at the end for the closing edge, and we must ensure
        # path edges are not forbidden.
        
        # Filter candidates: edge (cur, x) must not be forbidden
        valid_cand = []
        for x in cand:
            e = (cur, x) if cur < x else (x, cur)
            if e not in forbidden_edges:
                valid_cand.append(x)
        
        if not valid_cand:
            return False
            
        valid_cand.sort(key=lambda x: (sum(1 for y in adj[x] if not used[y]), rng.random()))

        for x in valid_cand:
            path.append(x)
            used[x] = True
            steps += 1
            if steps > max_steps:
                path.pop()
                used[x] = False
                return False
            if backtrack():
                return True
            path.pop()
            used[x] = False
        return False

    backtrack()
    return best_cycle


def _is_valid_cycles(cycles, all_edges):
    seen = set()
    vset = None
    for cyc in cycles:
        if len(cyc) != N:
            return False
        if len(set(cyc)) != N:
            return False
        if vset is None:
            vset = set(cyc)
        elif set(cyc) != vset:
            return False
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            if e not in all_edges:
                return False
            if e in seen:
                return False
            seen.add(e)
    return True


def _backtrack_cycles(target_count, used_edges, forbidden_edges, deadline, rng, adj, max_searches_per_level=100):
    """
    Recursively finds 'target_count' edge-disjoint Hamiltonian cycles.
    used_edges: edges already committed to previous cycles.
    forbidden_edges: edges to avoid in the *current* search to find a different cycle
                      if the previous choice led to a dead end.
    """
    if target_count == 0:
        return []
    
    # Check if enough edges remain
    if len(used_edges) + target_count * N > len(EDGE_SET):
        return None

    # Build adj for current state (excluding used_edges)
    # Note: adj passed in is BASE_ADJ. We need to filter by used_edges and forbidden_edges.
    # For efficiency, we pass the filtered adj or filter inside _find_hamiltonian?
    # _find_hamiltonian takes adj and forbidden_edges.
    # The adj should represent the graph minus used_edges.
    
    # Let's construct the effective adjacency for this level
    # This is expensive if done every time, but N=21 is small.
    eff_adj = [set() for _ in range(N)]
    for u in range(N):
        for v in BASE_ADJ[u]:
            e = (u, v) if u < v else (v, u)
            if e not in used_edges and e not in forbidden_edges:
                eff_adj[u].add(v)
                
    # Feasibility check on eff_adj
    for v in range(N):
        if len(eff_adj[v]) < 2:
            return None
    if not _check_connected(eff_adj, N):
        return None

    found_cycles = []
    searches_done = 0
    
    while searches_done < max_searches_per_level:
        # Find a cycle in the effective graph
        C = _find_hamiltonian(eff_adj, deadline, rng, forbidden_edges=set()) 
        # Note: _find_hamiltonian takes forbidden_edges, but we already filtered eff_adj.
        # So we pass empty forbidden to _find_hamiltonian, as the restriction is in eff_adj.
        
        if C is None:
            return None # No more cycles possible in this branch
        
        # Try to complete the packing with this cycle C
        new_used = used_edges | set()
        cyc_edges = set()
        for i in range(len(C)):
            a, b = C[i], C[(i + 1) % len(C)]
            e = (a, b) if a < b else (b, a)
            cyc_edges.add(e)
            
        new_used = used_edges | cyc_edges
        
        # Recurse
        rest = _backtrack_cycles(target_count - 1, new_used, set(), deadline, rng, BASE_ADJ, max_searches_per_level)
        
        if rest is not None:
            return [C] + rest
        
        # If it failed, we need to find a DIFFERENT cycle.
        # We do this by adding one edge from C to the 'forbidden' set for the next iteration.
        # This prevents C from being found again (since C must contain that edge).
        # It may allow other cycles that share other edges with C.
        
        # To be systematic, we can iterate through edges of C.
        # But we are in a while loop. We need to modify forbidden_edges to exclude C.
        # Let's pick the first edge of C to forbid.
        # However, if we just add one edge, we might cycle through many variations.
        # A better way: Add ALL edges of C to a temporary "blocked" set? No, that's too strong.
        # Adding just one edge is enough to distinguish C from itself.
        
        # To avoid infinite loops of blocking the same edge, we can cycle through the edges of C.
        # Or, simpler: Add the first edge of C to forbidden_edges.
        # But we need to ensure we don't retry the exact same state.
        # Since we add an edge to forbidden_edges, the state changes.
        
        # Which edge to add?
        # Let's add the edge (C[0], C[1]).
        a, b = C[0], C[1]
        e_to_block = (a, b) if a < b else (b, a)
        
        # We need to update forbidden_edges to include this edge.
        # But forbidden_edges is passed by value (set). We create a new set.
        # Wait, the while loop logic:
        # We want to try the next best cycle.
        # By adding e_to_block to forbidden_edges, we force the next _find_hamiltonian
        # to find a cycle that does NOT use e_to_block.
        # This effectively skips C (and any other cycle using e_to_block, which is fine).
        
        forbidden_edges = forbidden_edges | {e_to_block}
        searches_done += 1
        
        # Optional: If we have tried many, we might want to randomize which edge to block
        # to explore different branches. But deterministic is safer for reproducibility.
        
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    rng = random.Random(12345)
    
    # Try to find 3 cycles first (max possible is 3 since 70 edges / 21 = 3.33)
    # If 3 fails, try 2, then 1.
    
    # The backtracking function handles the "try different cycles" logic internally.
    # We just call it for target_count = 3.
    
    result = _backtrack_cycles(3, set(), set(), deadline, rng, BASE_ADJ, max_searches_per_level=50)
    
    if result is not None:
        if _is_valid_cycles(result, EDGE_SET):
            return [list(c) for c in result]
            
    # Fallback to 2
    result = _backtrack_cycles(2, set(), set(), deadline, rng, BASE_ADJ, max_searches_per_level=50)
    if result is not None:
        if _is_valid_cycles(result, EDGE_SET):
            return [list(c) for c in result]
            
    # Fallback to 1
    result = _backtrack_cycles(1, set(), set(), deadline, rng, BASE_ADJ, max_searches_per_level=50)
    if result is not None:
        if _is_valid_cycles(result, EDGE_SET):
            return [list(c) for c in result]
            
    # Absolute fallback
    return [list(range(N))]
# EVOLVE-BLOCK-END
