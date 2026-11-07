import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_packing(cycles, all_edges_set):
    """
    Verify that all cycles are valid, edge-disjoint, and use the same vertex set.
    Returns True if valid, False otherwise.
    """
    if not cycles:
        return False
    
    n = P['n']
    used_edges = set()
    target_vertex_set = None
    
    for cycle in cycles:
        if not isinstance(cycle, (list, tuple)):
            return False
        if len(cycle) < 3:
            return False
        
        for v in cycle:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= n:
                return False
        
        if len(set(cycle)) != len(cycle):
            return False
        
        current_vset = set(cycle)
        if target_vertex_set is None:
            target_vertex_set = current_vset
        else:
            if current_vset != target_vertex_set:
                return False
        
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in all_edges_set:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n = P['n']
    all_edges = P['edges']
    all_edges_set = set()
    for u, v in all_edges:
        all_edges_set.add(_edge_key(u, v))

    # The graph is a 6-regular circulant on 21 vertices with 63 edges.
    # Theoretical max edge-disjoint Hamiltonian cycles = 63 // 21 = 3.
    # We construct 3 using circulant steps 1, 5, and 19 (= -2 mod 21).
    # These steps are pairwise distinct modulo 21 and their edge sets partition E.

    def make_cycle(step):
        return [(i * step) % n for i in range(n)]

    # Primary: hardcoded optimal packing
    cycles = [
        make_cycle(1),
        make_cycle(5),
        make_cycle(19)
    ]

    # Ensure strict Python int types
    validated_cycles = [[int(v) for v in c] for c in cycles]

    if _verify_packing(validated_cycles, all_edges_set):
        return validated_cycles

    # Fallback: if the hardcoded solution somehow fails (should not happen),
    # try a quick backtracking search to find any valid packing.
    if time.time() < deadline:
        from collections import defaultdict
        import random

        def _find_one_hamiltonian(remaining_edge_set, rng):
            n_local = P['n']
            adj = defaultdict(set)
            for ek in remaining_edge_set:
                u, v = ek
                adj[u].add(v)
                adj[v].add(u)

            valid_starts = [v for v in range(n_local) if len(adj[v]) >= 2]
            if not valid_starts:
                return None

            for _ in range(200):
                start = rng.choice(valid_starts)

                def backtrack(path, visited):
                    if len(path) == n_local:
                        if start in adj[path[-1]]:
                            return path[:]
                        return None
                    current = path[-1]
                    candidates = []
                    for nb in adj[current]:
                        if nb not in visited:
                            onward = len([x for x in adj[nb] if x not in visited and x != current])
                            candidates.append((onward, nb))
                    if not candidates:
                        return None
                    candidates.sort(key=lambda x: x[0])
                    k = min(len(candidates), 3)
                    top = candidates[:k]
                    rng.shuffle(top)
                    rest = candidates[k:]
                    for _, nb in top + rest:
                        result = backtrack(path + [nb], visited | {nb})
                        if result is not None:
                            return result
                    return None

                cycle = backtrack([start], {start})
                if cycle is not None:
                    # Verify
                    valid = True
                    for i in range(n_local):
                        u, v = cycle[i], cycle[(i + 1) % n_local]
                        ek = _edge_key(u, v)
                        if ek not in remaining_edge_set:
                            valid = False
                            break
                    if valid:
                        return [int(x) for x in cycle]
            return None

        best = []
        for seed in range(500):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            remaining = set(all_edges_set)
            cycles_found = []
            for _ in range(3):
                if time.time() > deadline:
                    break
                c = _find_one_hamiltonian(remaining, rng)
                if c is None:
                    break
                for i in range(len(c)):
                    u, v = c[i], c[(i + 1) % len(c)]
                    remaining.discard(_edge_key(u, v))
                cycles_found.append(c)
            if len(cycles_found) > len(best):
                best = cycles_found
                if len(best) == 3:
                    break

        if best and _verify_packing(best, all_edges_set):
            return best

    return []

# EVOLVE-BLOCK-END
