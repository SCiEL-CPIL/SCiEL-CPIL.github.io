"""Packing trees T1..T20 into K20 via greedy matching-based embedding."""
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def _bipartite_matching(left_nodes, right_nodes, adj):
    """Find maximum matching in bipartite graph. adj[u] = list of v's."""
    match_r = {}
    for u in left_nodes:
        visited = set()
        def dfs(u):
            for v in adj.get(u, []):
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_r or dfs(match_r[v]):
                    match_r[v] = u
                    return True
            return False
        dfs(u)
    return {u: v for v, u in match_r.items()}

def construct():
    trees = instance_trees(20, 611)
    n = 20
    used_edges = set()  # frozenset of (a,b) with a < b
    result = [None] * 20
    
    for k in range(1, n + 1):
        edges = trees[k - 1]
        if not edges:
            result[k - 1] = []
            continue
        
        # Find a bijection from tree vertices {0..k-1} to host vertices {0..19}
        # such that all tree edges map to unused host edges
        # Use backtracking with heuristic ordering
        
        best_map = None
        # Try random permutations with a limit
        rng = Random(42 + k)
        host_verts = list(range(n))
        
        # Heuristic: try to find embedding via constraint propagation
        # Order tree edges by difficulty (higher degree endpoints first)
        deg = [0] * k
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        
        # Sort vertices by degree descending for assignment order
        vertex_order = sorted(range(k), key=lambda v: -deg[v])
        
        # Backtracking search
        def backtrack(assigned, pos):
            nonlocal best_map
            if best_map is not None:
                return True
            if pos == k:
                best_map = dict(assigned)
                return True
            
            tv = vertex_order[pos]
            # Try each unused host vertex
            for hv in host_verts:
                if hv in assigned.values():
                    continue
                # Check all edges from tv to already-assigned vertices
                valid = True
                for a, b in edges:
                    if a == tv and b in assigned:
                        e = frozenset((hv, assigned[b]))
                        if e in used_edges:
                            valid = False
                            break
                    elif b == tv and a in assigned:
                        e = frozenset((hv, assigned[a]))
                        if e in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                assigned[tv] = hv
                if backtrack(assigned, pos + 1):
                    return True
                del assigned[tv]
            return False
        
        assigned = {}
        if not backtrack(assigned, 0):
            # Try randomized restarts
            for attempt in range(50):
                rng2 = Random(1000 + k * 100 + attempt)
                host_order = list(range(n))
                rng2.shuffle(host_order)
                # Redefine backtrack with shuffled host order
                def backtrack2(assigned, pos):
                    nonlocal best_map
                    if best_map is not None:
                        return True
                    if pos == k:
                        best_map = dict(assigned)
                        return True
                    tv = vertex_order[pos]
                    for hv in host_order:
                        if hv in assigned.values():
                            continue
                        valid = True
                        for a, b in edges:
                            if a == tv and b in assigned:
                                e = frozenset((hv, assigned[b]))
                                if e in used_edges:
                                    valid = False
                                    break
                            elif b == tv and a in assigned:
                                e = frozenset((hv, assigned[a]))
                                if e in used_edges:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        assigned[tv] = hv
                        if backtrack2(assigned, pos + 1):
                            return True
                        del assigned[tv]
                    return False
                assigned = {}
                if backtrack2(assigned, 0):
                    break
        
        if best_map is None:
            # Fallback: skip this tree
            result[k - 1] = None
            continue
        
        # Convert to list format: position i is image of vertex i
        mapping = [best_map[i] for i in range(k)]
        result[k - 1] = [int(x) for x in mapping]
        
        # Mark edges as used
        for a, b in edges:
            e = frozenset((best_map[a], best_map[b]))
            used_edges.add(e)
    
    return result
