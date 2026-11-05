"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 20  # seconds to spare before the 150s kill

# Precompute Q3 structure
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

_Q3_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for bit in (1, 2, 4):
        j = i ^ bit
        if j > i:
            _Q3_NEIGHBORS[i].append(j)
            _Q3_NEIGHBORS[j].append(i)


def _has_q3_fast(adj_bits):
    """
    Check if graph with adjacency bitmask contains Q3.
    Uses backtracking with pruning.
    """
    assigned = [-1] * 8
    used_mask = 0

    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        for gv in range(_N):
            if (used_mask >> gv) & 1:
                continue
            ok = True
            for pv in range(qv):
                if pv in _Q3_NEIGHBORS[qv]:
                    if not (adj_bits[gv] >> assigned[pv]) & 1:
                        ok = False
                        break
            if ok:
                assigned[qv] = gv
                used_mask |= (1 << gv)
                if backtrack(qv + 1):
                    return True
                used_mask &= ~(1 << gv)
                assigned[qv] = -1
        return False

    return backtrack(0)


def _build_adj_bits(edge_list):
    """Build adjacency bitmask representation from edge list."""
    adj = [0] * _N
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _build_adj_matrix(edge_list):
    """Build adjacency matrix (NumPy) from edge list."""
    A = np.zeros((_N, _N), dtype=np.int64)
    for u, v in edge_list:
        A[u, v] = 1
        A[v, u] = 1
    return A


def _count_4cycles(A):
    """
    Count the number of 4-cycles in the graph using trace(A^4).
    trace(A^4) = sum of diagonal entries of A^4
    This counts closed walks of length 4.
    Number of 4-cycles = (trace(A^4) - 4*n*(avg_degree^2) + ...) 
    Actually, a simpler formula:
    The number of (not necessarily induced) 4-cycles can be computed as:
    For each pair of vertices (u,v) not adjacent, count common neighbors.
    But using A^4: trace(A^4) counts closed walks of length 4.
    Each 4-cycle contributes 8 to trace(A^4) (4 starting points * 2 directions).
    Additionally, walks that go back and forth on edges also contribute.
    
    A cleaner approach: count 4-cycles directly.
    A 4-cycle is u-v-w-x-u where all four are distinct and edges exist.
    We can count by: for each pair (u,v), count common neighbors, then adjust.
    
    Actually, let's just use a direct count:
    Number of 4-cycles = (1/8) * sum over all ordered pairs (u,v) of C(common_neighbors(u,v), 2)
    where we count each 4-cycle 8 times (4 vertices * 2 directions).
    
    But for the spectral heuristic, we just need a proxy. Let's use:
    For edge (u,v), the "4-cycle contribution" is the number of common neighbors of u and v
    that could form 4-cycles through this edge.
    
    Simpler: use A^2 diagonal. (A^2)[u,v] = number of common neighbors of u and v.
    The number of 4-cycles through edge (u,v) is related to (A^2)[u,v].
    Specifically, if we remove edge (u,v), the change in 4-cycle count is approximately
    -(A^2)[u,v] * (something).
    
    Let me use a direct approach: for each edge (u,v), compute how many 4-cycles
    contain this edge. A 4-cycle containing edge (u,v) has the form u-v-w-x-u
    where w is a neighbor of v (not u), x is a neighbor of u (not v), and w-x is an edge.
    So the count is: for each w in N(v) \ {u}, for each x in N(u) \ {v}, check if w-x is edge.
    This is O(degree^2) per edge, which is fine for n=16.
    """
    # Direct count of 4-cycles containing each edge
    # For efficiency with n=16, we can precompute common neighbors
    A2 = A @ A  # A^2, A2[u,v] = number of common neighbors of u and v
    
    # For edge (u,v), number of 4-cycles through it:
    # We need w adjacent to v, x adjacent to u, w != u, x != v, w-x adjacent
    # Count = sum over w in N(v)\{u} of |N(w) ∩ N(u) \ {v}|
    # But this is getting complex. Let me just use a simpler proxy.
    
    # Proxy: for edge (u,v), the "4-cycle penalty" is (A^2)[u,v]
    # because each common neighbor of u and v forms a triangle, and removing the edge
    # breaks potential 4-cycles.
    
    # Actually, let me count 4-cycles properly.
    # A 4-cycle is a set {u,v,w,x} with edges uv, vw, wx, xu (cyclic).
    # Count all 4-cycles: for each 4-set, check if it forms a cycle.
    # For n=16, C(16,4) = 1820, very fast.
    
    from itertools import combinations
    
    count = 0
    for quad in combinations(range(_N), 4):
        u, v, w, x = quad
        # Check all 3 ways to form a 4-cycle on these 4 vertices
        # Cycle 1: u-v-w-x-u
        if A[u,v] and A[v,w] and A[w,x] and A[x,u]:
            count += 1
        # Cycle 2: u-v-x-w-u
        if A[u,v] and A[v,x] and A[x,w] and A[w,u]:
            count += 1
        # Cycle 3: u-w-v-x-u
        if A[u,w] and A[w,v] and A[v,x] and A[x,u]:
            count += 1
    
    return count


def _edge_4cycle_contribution(A, u, v):
    """
    Count the number of 4-cycles that contain edge (u,v).
    A 4-cycle containing edge (u,v) has form u-v-w-x-u where
    w is neighbor of v (w != u), x is neighbor of u (x != v), and w-x is edge.
    """
    count = 0
    for w in range(_N):
        if w == u or w == v:
            continue
        if A[v, w]:
            for x in range(_N):
                if x == u or x == v or x == w:
                    continue
                if A[u, x] and A[w, x]:
                    count += 1
    return count


def _has_q3_exact(adj_bits):
    """Exact Q3 check using backtracking."""
    return _has_q3_fast(adj_bits)


def construct():
    start_time = time.time()
    deadline = start_time + (_DEADLINE_MARGIN)  # 130s total, 20s margin
    
    # Start with K16
    edges = [[u, v] for u in range(_N) for v in range(u + 1, _N)]
    adj_bits = _build_adj_bits(edges)
    A = _build_adj_matrix(edges)
    
    # Phase 1: Spectral heuristic - remove edges to minimize 4-cycles
    # While graph has Q3, remove the edge with highest 4-cycle contribution
    removed = set()
    
    while time.time() < deadline:
        # Check if Q3-free
        if not _has_q3_exact(adj_bits):
            break
        
        # Find the edge with the highest 4-cycle contribution
        best_edge = None
        best_contrib = -1
        for u in range(_N):
            for v in range(u + 1, _N):
                if (u, v) in removed or (v, u) in removed:
                    continue
                if not (adj_bits[u] >> v) & 1:
                    continue
                contrib = _edge_4cycle_contribution(A, u, v)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_edge = (u, v)
        
        if best_edge is None or best_contrib <= 0:
            # No edge to remove that helps, break
            break
        
        u, v = best_edge
        # Remove edge (u, v)
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
        A[u, v] = 0
        A[v, u] = 0
        removed.add((u, v))
        edges.remove([u, v]) if [u, v] in edges else edges.remove([v, u])
        
        # Check time
        if time.time() > deadline:
            break
    
    # Phase 2: If still not Q3-free, do more aggressive removal
    if time.time() < deadline:
        while _has_q3_exact(adj_bits):
            if time.time() > deadline:
                break
            # Remove a random edge that's part of many 4-cycles
            # Try all remaining edges, pick one with highest 4-cycle contribution
            best_edge = None
            best_contrib = -1
            for u in range(_N):
                for v in range(u + 1, _N):
                    if not (adj_bits[u] >> v) & 1:
                        continue
                    contrib = _edge_4cycle_contribution(A, u, v)
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            A[u, v] = 0
            A[v, u] = 0
            if [u, v] in edges:
                edges.remove([u, v])
            elif [v, u] in edges:
                edges.remove([v, u])
    
    # Phase 3: Try to add back edges greedily
    if time.time() < deadline - 10:
        # Try to add back removed edges
        for (u, v) in list(removed):
            if time.time() > deadline - 5:
                break
            # Try adding edge (u, v)
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            A[u, v] = 1
            A[v, u] = 1
            
            if not _has_q3_exact(adj_bits):
                # Keep the edge
                edges.append([u, v])
                removed.discard((u, v))
            else:
                # Remove it again
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                A[u, v] = 0
                A[v, u] = 0
    
    # Phase 4: Try to add any other missing edges
    if time.time() < deadline - 5:
        for u in range(_N):
            for v in range(u + 1, _N):
                if time.time() > deadline - 2:
                    break
                if (adj_bits[u] >> v) & 1:
                    continue
                # Try adding edge (u, v)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
                if not _has_q3_exact(adj_bits):
                    edges.append([u, v])
                else:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
    
    return edges
# EVOLVE-BLOCK-END
