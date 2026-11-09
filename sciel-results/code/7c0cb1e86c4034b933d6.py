"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import math

# EVOLVE-BLOCK-START

# Precompute all Q3 subgraphs (as frozensets of edge indices) for fast checking
# A Q3 is a 3-cube: 8 vertices, 12 edges.
# We enumerate all 8-subsets of {0..15} and check if they form a Q3.

# Edge list for K16
_ALL_EDGES = [(i, j) for i in range(16) for j in range(i+1, 16)]
N_EDGES = len(_ALL_EDGES)  # 120

# Map edge to index
_EDGE_INDEX = {}
for _idx, (_u, _v) in enumerate(_ALL_EDGES):
    _EDGE_INDEX[(_u, _v)] = _idx
    _EDGE_INDEX[(_v, _u)] = _idx

# Precompute all Q3 edge-sets
# A Q3 has 8 vertices and 12 edges. The 8 vertices form a 3-cube.
# We need to find all subsets of 8 vertices that induce exactly a Q3 (12 edges, each vertex degree 3).

_Q3_EDGE_SETS = []
_Q3_VERTEX_SETS = []

# Generate all 8-combinations of 16 vertices
# For each, check if the induced subgraph is exactly Q3 (12 edges, all degrees 3)
for _verts in itertools.combinations(range(16), 8):
    _vset = set(_verts)
    # Count edges within this set
    _edges_in_set = []
    for _i, _a in enumerate(_verts):
        for _b in _verts[_i+1:]:
            if (_a, _b) in _EDGE_INDEX:
                _edges_in_set.append((_a, _b))
    if len(_edges_in_set) != 12:
        continue
    # Check all vertices have degree 3 within the set
    _deg = {v: 0 for v in _verts}
    for _a, _b in _edges_in_set:
        _deg[_a] += 1
        _deg[_b] += 1
    if all(_deg[v] == 3 for v in _verts):
        # Also verify it's connected and isomorphic to Q3
        # Q3 is bipartite with parts of size 4 each
        # Check bipartiteness
        _adj = {v: set() for v in _verts}
        for _a, _b in _edges_in_set:
            _adj[_a].add(_b)
            _adj[_b].add(_a)
        # BFS for bipartite check
        _color = {}
        _is_bipartite = True
        _start = _verts[0]
        _color[_start] = 0
        _queue = [_start]
        while _queue:
            _u = _queue.pop(0)
            for _w in _adj[_u]:
                if _w not in _color:
                    _color[_w] = 1 - _color[_u]
                    _queue.append(_w)
                elif _color[_w] == _color[_u]:
                    _is_bipartite = False
                    break
        if not _is_bipartite:
            continue
        # Check both parts have size 4
        _part0 = [v for v in _verts if _color[v] == 0]
        _part1 = [v for v in _verts if _color[v] == 1]
        if len(_part0) != 4 or len(_part1) != 4:
            continue
        # Store
        _edge_idx_set = frozenset(_EDGE_INDEX[e] for e in _edges_in_set)
        _Q3_EDGE_SETS.append(_edge_idx_set)
        _Q3_VERTEX_SETS.append(frozenset(_verts))

# Deduplicate (same edge set might appear once per vertex permutation, but we use frozenset of edge indices)
_Q3_EDGE_SETS = list(set(_Q3_EDGE_SETS))
_N_Q3 = len(_Q3_EDGE_SETS)

# For each edge, precompute which Q3s contain it
_EDGE_TO_Q3S = [[] for _ in range(N_EDGES)]
for _qi, _q3 in enumerate(_Q3_EDGE_SETS):
    for _ei in _q3:
        _EDGE_TO_Q3S[_ei].append(_qi)

# For each Q3, precompute the set of its edges as a bitmask (using Python int)
_Q3_MASKS = [0] * _N_Q3
for _qi, _q3 in enumerate(_Q3_EDGE_SETS):
    _mask = 0
    for _ei in _q3:
        _mask |= (1 << _ei)
    _Q3_MASKS[_qi] = _mask

def _has_q3(edge_mask):
    """Check if a set of edges (as bitmask) contains any Q3."""
    for _qi in range(_N_Q3):
        if (edge_mask & _Q3_MASKS[_qi]) == _Q3_MASKS[_qi]:
            return True
    return False

def _count_q3s(edge_mask):
    """Count how many Q3s are present in the edge set."""
    cnt = 0
    for _qi in range(_N_Q3):
        if (edge_mask & _Q3_MASKS[_qi]) == _Q3_MASKS[_qi]:
            cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with a good initial solution
    # K3,13 has 39 edges. Let's try to build a denser Q3-free graph.
    
    # Strategy: Greedy construction + local search (simulated annealing style)
    
    # Greedy: add edges in order of "least likely to create Q3"
    # Sort edges by some heuristic
    random.seed(42)
    
    # Initial: try to build a graph by adding edges one by one
    # Use a random order, skip if it creates a Q3
    edge_order = list(range(N_EDGES))
    random.shuffle(edge_order)
    
    current_mask = 0
    for ei in edge_order:
        new_mask = current_mask | (1 << ei)
        if not _has_q3(new_mask):
            current_mask = new_mask
    
    best_mask = current_mask
    best_count = bin(current_mask).count('1')
    
    # Local search: try to add edges, or swap edges
    # Simulated annealing style
    T = 50.0  # initial temperature
    cooling = 0.995
    min_T = 0.01
    
    # Precompute for each edge, the Q3s it participates in
    # For speed, when considering adding edge e, check only Q3s containing e
    
    while time.time() < deadline:
        if T < min_T:
            break
        
        # Try a random move: add an edge, remove an edge, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random edge
            ei = random.randint(0, N_EDGES - 1)
            if (current_mask >> ei) & 1:
                continue
            new_mask = current_mask | (1 << ei)
            # Check if new_mask has a Q3: only need to check Q3s containing ei
            has = False
            for qi in _EDGE_TO_Q3S[ei]:
                if (new_mask & _Q3_MASKS[qi]) == _Q3_MASKS[qi]:
                    has = True
                    break
            if not has:
                current_mask = new_mask
                new_count = bin(current_mask).count('1')
                if new_count > best_count:
                    best_mask = current_mask
                    best_count = new_count
            else:
                # Rejection
                pass
        elif move_type < 0.7:
            # Try to remove a random edge (to escape local optima)
            ei = random.randint(0, N_EDGES - 1)
            if not ((current_mask >> ei) & 1):
                continue
            new_mask = current_mask & ~(1 << ei)
            # Always accept removal in SA if it doesn't hurt too much
            new_count = bin(new_mask).count('1')
            delta = new_count - bin(current_mask).count('1')
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                if new_count > best_count:
                    best_mask = current_mask
                    best_count = new_count
        else:
            # Swap: remove one edge, add another
            ei_remove = random.randint(0, N_EDGES - 1)
            ei_add = random.randint(0, N_EDGES - 1)
            if ei_remove == ei_add:
                continue
            if not ((current_mask >> ei_remove) & 1):
                continue
            if (current_mask >> ei_add) & 1:
                continue
            
            new_mask = (current_mask & ~(1 << ei_remove)) | (1 << ei_add)
            # Check Q3s: need to check Q3s containing ei_add (newly added)
            # and Q3s containing ei_remove (might have been broken)
            # Simpler: just check all Q3s containing ei_add
            has = False
            for qi in _EDGE_TO_Q3S[ei_add]:
                if (new_mask & _Q3_MASKS[qi]) == _Q3_MASKS[qi]:
                    has = True
                    break
            if not has:
                current_mask = new_mask
                new_count = bin(current_mask).count('1')
                if new_count > best_count:
                    best_mask = current_mask
                    best_count = new_count
        
        T *= cooling
    
    # Convert best_mask to edge list
    result = []
    for ei in range(N_EDGES):
        if (best_mask >> ei) & 1:
            result.append(list(_ALL_EDGES[ei]))
    
    return result

# EVOLVE-BLOCK-END
