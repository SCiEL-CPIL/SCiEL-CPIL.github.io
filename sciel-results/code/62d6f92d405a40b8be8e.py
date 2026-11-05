import time
import random

N = 40

def _validate(edges, cycles):
    n = N
    edge_set = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))

    if len(cycles) != n - 2:
        return False

    for i, cycle in enumerate(cycles):
        L = i + 3
        if len(cycle) != L:
            return False
        if len(set(cycle)) != L:
            return False
        for j in range(L):
            u = cycle[j]
            v = cycle[(j + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return False
    return True

def _build_adj(edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # --- Strategy: Optimize Fan Graph ---
    # Fan Graph: Path 1-2-...-39 (38 edges) + Spokes 0-s for s in S.
    # Total edges: 38 + |S|.
    # Requirement: S-S covers {1, ..., 38}.
    
    def covers_all(S):
        S_set = set(S)
        covered = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = abs(S[i] - S[j])
                if d <= 38:
                    covered.add(d)
        return covered == set(range(1, 39))

    def find_min_S(time_limit=5.0):
        # Greedy Init
        S = [1, 39]
        covered = {38}
        while len(covered) < 38:
            missing = [d for d in range(1, 39) if d not in covered]
            if not missing:
                break
            best_x = -1
            best_new = -1
            for x in range(1, 40):
                if x in S:
                    continue
                new_covered = set(covered)
                for y in S:
                    d = abs(x - y)
                    if d <= 38:
                        new_covered.add(d)
                new_only = new_covered - covered
                if len(new_only) > best_new:
                    best_new = len(new_only)
                    best_x = x
            if best_x == -1:
                break
            S.append(best_x)
            for y in S:
                d = abs(best_x - y)
                if d <= 38:
                    covered.add(d)
        
        best_S = S[:]
        best_size = len(best_S)
        current_S = S[:]
        
        start = time.time()
        while time.time() - start < time_limit:
            # Try remove
            removed = False
            if len(current_S) > 2:
                idx = random.randint(0, len(current_S)-1)
                test_S = current_S[:idx] + current_S[idx+1:]
                if covers_all(test_S):
                    current_S = test_S
                    if len(current_S) < best_size:
                        best_size = len(current_S)
                        best_S = current_S[:]
                    removed = True
            
            # Try swap
            if not removed and len(current_S) > 0:
                idx = random.randint(0, len(current_S)-1)
                old_val = current_S[idx]
                new_val = random.randint(1, 39)
                if new_val not in current_S:
                    test_S = current_S[:]
                    test_S[idx] = new_val
                    if covers_all(test_S):
                        current_S = test_S
                        if len(current_S) < best_size:
                            best_size = len(current_S)
                            best_S = current_S[:]
        
        return best_S

    # Run S optimization
    S_opt = find_min_S()
    
    # Build Fan Graph
    path_edges = [(i, i + 1) for i in range(1, N - 1)]
    spoke_edges = [(0, s) for s in S_opt]
    edges = path_edges + spoke_edges
    
    # Build Cycles
    cycles = []
    S_set = set(S_opt)
    valid = True
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        else:
            diff = L - 2
            found = False
            for u in S_set:
                v = u + diff
                if v in S_set:
                    cycle_verts = [0] + list(range(u, v + 1))
                    cycles.append(cycle_verts)
                    found = True
                    break
            if not found:
                valid = False
                break
    
    if valid and _validate(edges, cycles):
        return {'edges': edges, 'cycles': cycles}
        
    # Fallback: Ham + Chords (59 edges)
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    chord_edges = [(0, d) for d in range(2, 21)]
    edges = ham_edges + chord_edges
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        elif L <= 21:
            cycles.append(list(range(L)))
        else:
            end_vertex = N + 1 - L
            cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
            
    return {'edges': edges, 'cycles': cycles}
