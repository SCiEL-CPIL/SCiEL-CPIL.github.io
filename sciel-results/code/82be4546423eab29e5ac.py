"""Improved program for cochromatic partition of a pinned random graph."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

# Precompute adjacency sets for fast clique/independent set checks
_N = P['n']
_edges = P['edges']
_adj = [set() for _ in range(_N)]
_nonadj = [set() for _ in range(_N)]
for _u, _v in _edges:
    _adj[_u].add(_v)
    _adj[_v].add(_u)
for _i in range(_N):
    for _j in range(_N):
        if _i != _j and _j not in _adj[_i]:
            _nonadj[_i].add(_j)

def _is_clique(vertices):
    """Check if all pairs in vertices are adjacent."""
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if vertices[j] not in _adj[vertices[i]]:
                return False
    return True

def _is_independent(vertices):
    """Check if no pairs in vertices are adjacent."""
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if vertices[j] in _adj[vertices[i]]:
                return False
    return True

def _max_clique_size(subset):
    """Find maximum clique size within subset using Bron-Kerbosch."""
    if not subset:
        return 0
    subset_set = set(subset)
    best = [0]
    
    def bron_kerbosch(r, p, x):
        if not p:
            if len(r) > best[0]:
                best[0] = len(r)
            return
        # Pivot selection
        if p:
            pivot = next(iter(p))
            candidates = p - _adj[pivot] - {pivot}
        else:
            candidates = p
        for v in list(candidates):
            bron_kerbosch(r | {v}, p & _adj[v], x & _adj[v])
            p.remove(v)
            x.add(v)
    
    bron_kerbosch(set(), subset_set, set())
    return best[0]

def _max_independent_size(subset):
    """Find maximum independent set size within subset (equivalent to max clique in complement)."""
    if not subset:
        return 0
    subset_set = set(subset)
    best = [0]
    
    def bron_kerbosch(r, p, x):
        if not p:
            if len(r) > best[0]:
                best[0] = len(r)
            return
        if p:
            pivot = next(iter(p))
            candidates = p - _nonadj[pivot] - {pivot}
        else:
            candidates = p
        for v in list(candidates):
            bron_kerbosch(r | {v}, p & _nonadj[v], x & _nonadj[v])
            p.remove(v)
            x.add(v)
    
    bron_kerbosch(set(), subset_set, set())
    return best[0]

def _cochromatic_number_bruteforce(n, edges_set, deadline):
    """
    Find minimum cochromatic number by trying k=1,2,3,... and checking
    if vertices can be partitioned into k sets, each inducing a clique or independent set.
    Uses backtracking with pruning.
    """
    # Precompute for each subset whether it's a clique or independent set
    # Since n is small (14), we can use bitmask representation
    # But for clarity, we'll use sets of vertices
    
    all_vertices = list(range(n))
    
    def _can_partition_into_k(k):
        """Check if we can partition all_vertices into k parts, each clique or independent."""
        if k == 0:
            return not all_vertices
        if k == 1:
            return _is_clique(all_vertices) or _is_independent(all_vertices)
        
        # Backtracking: assign each vertex to one of k parts
        # parts[i] is a set of vertices in part i
        parts = [set() for _ in range(k)]
        
        def _try_assign(idx):
            if idx == len(all_vertices):
                return True
            v = all_vertices[idx]
            for i in range(k):
                # Check if adding v to part i keeps it clique or independent
                new_part = parts[i] | {v}
                if len(new_part) == 1:
                    parts[i].add(v)
                    if _try_assign(idx + 1):
                        return True
                    parts[i].remove(v)
                else:
                    if _is_clique(new_part) or _is_independent(new_part):
                        parts[i].add(v)
                        if _try_assign(idx + 1):
                            return True
                        parts[i].remove(v)
            return False
        
        return _try_assign(0)
    
    # Try k = 1, 2, 3, ...
    for k in range(1, n + 1):
        if time.time() > deadline:
            return None
        if _can_partition_into_k(k):
            # Now we need to find the actual partition
            # Use the same backtracking but record the partition
            parts = [set() for _ in range(k)]
            
            def _find_partition(idx):
                if idx == len(all_vertices):
                    return True
                v = all_vertices[idx]
                for i in range(k):
                    new_part = parts[i] | {v}
                    if len(new_part) == 1 or _is_clique(new_part) or _is_independent(new_part):
                        parts[i].add(v)
                        if _find_partition(idx + 1):
                            return True
                        parts[i].remove(v)
                return False
            
            if _find_partition(0):
                # Convert to list of lists
                result = [sorted(list(p)) for p in parts if p]
                return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s safety margin
    
    # Try to find optimal cochromatic number
    result = _cochromatic_number_bruteforce(_N, set(map(tuple, _edges)), deadline)
    
    if result is not None:
        # Ensure all elements are Python int and lists
        cleaned = []
        for part in result:
            cleaned.append([int(v) for v in part])
        return cleaned
    else:
        # Fallback: greedy approach (same as seed)
        es = {tuple(e) for e in P['edges']}
        parts = []
        for v in range(P['n']):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
# EVOLVE-BLOCK-END
