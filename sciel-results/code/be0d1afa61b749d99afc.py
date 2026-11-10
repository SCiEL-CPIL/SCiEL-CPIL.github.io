"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _count_bicolored_cycles(coloring, k, adj):
    """Count total bicolored cycles across all color pairs."""
    n = len(coloring)
    total = 0
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[ry] = rx
                return False
            cycles = 0
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v > u and coloring[v] in (c1, c2):
                        if union(u, v):
                            cycles += 1
            total += cycles
    return total

def _is_valid(coloring, k, adj):
    n = len(coloring)
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    return _count_bicolored_cycles(coloring, k, adj) == 0

def _local_search_3color(adj, n, max_time=70.0):
    """Local search for valid 3-coloring."""
    start_time = time.time()
    best = None
    best_cycles = float('inf')
    
    for attempt in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with pattern
        if attempt % 3 == 0:
            coloring = [i % 3 for i in range(n)]
        elif attempt % 3 == 1:
            coloring = [(i * 7) % 3 for i in range(n)]
        else:
            coloring = [random.randint(0, 2) for _ in range(n)]
        
        cycles = _count_bicolored_cycles(coloring, 3, adj)
        
        # Local search
        stall = 0
        for step in range(200):
            if time.time() - start_time > max_time:
                break
            if cycles == 0:
                # Verify proper coloring
                valid = True
                for u, v in P['edges']:
                    if coloring[u] == coloring[v]:
                        valid = False
                        break
                if valid:
                    best = list(coloring)
                    return best
                # Fix proper coloring violations
                for v in range(n):
                    conflict = False
                    for nb in adj[v]:
                        if coloring[nb] == coloring[v]:
                            conflict = True
                            break
                    if conflict:
                        for c in range(3):
                            if c != coloring[v]:
                                ok = True
                                for nb in adj[v]:
                                    if coloring[nb] == c:
                                        ok = False
                                        break
                                if ok:
                                    coloring[v] = c
                                    break
                cycles = _count_bicolored_cycles(coloring, 3, adj)
                continue
            
            # Find best move
            best_move = None
            best_delta = float('inf')
            for v in range(n):
                old_c = coloring[v]
                for new_c in range(3):
                    if new_c == old_c:
                        continue
                    # Check neighbor conflict
                    if any(coloring[nb] == new_c for nb in adj[v]):
                        continue
                    # Estimate cycle change
                    coloring[v] = new_c
                    new_cycles = _count_bicolored_cycles(coloring, 3, adj)
                    coloring[v] = old_c
                    delta = new_cycles - cycles
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (v, new_c)
            
            if best_move is not None and best_delta < 0:
                v, new_c = best_move
                coloring[v] = new_c
                cycles += best_delta
                stall = 0
            else:
                # Random kick
                v = random.randint(0, n - 1)
                old_c = coloring[v]
                new_c = random.choice([c for c in range(3) if c != old_c])
                if not any(coloring[nb] == new_c for nb in adj[v]):
                    coloring[v] = new_c
                    cycles = _count_bicolored_cycles(coloring, 3, adj)
                stall += 1
                if stall > 50:
                    break
        
        if cycles < best_cycles:
            best_cycles = cycles
            best = list(coloring)
    
    if best is not None:
        # Final verification
        k = 3
        if _is_valid(best, k, adj):
            return best
    return None

def construct():
    start_time = time.time()
    deadline = 85.0
    n = P['n']
    adj = _build_adj()
    
    # Try 3-coloring
    coloring = _local_search_3color(adj, n, max_time=min(deadline - 10.0, 60.0))
    if coloring is not None:
        return [int(c) for c in coloring]
    
    # Try 4-coloring
    best4 = None
    for attempt in range(3000):
        if time.time() - start_time > deadline:
            break
        coloring = [i % 4 for i in range(n)] if attempt % 2 == 0 else [random.randint(0, 3) for _ in range(n)]
        cycles = _count_bicolored_cycles(coloring, 4, adj)
        stall = 0
        for step in range(100):
            if time.time() - start_time > deadline:
                break
            if cycles == 0:
                valid = True
                for u, v in P['edges']:
                    if coloring[u] == coloring[v]:
                        valid = False
                        break
                if valid:
                    best4 = list(coloring)
                    break
                break
            best_move = None
            best_delta = float('inf')
            for v in range(n):
                old_c = coloring[v]
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    if any(coloring[nb] == new_c for nb in adj[v]):
                        continue
                    coloring[v] = new_c
                    new_cycles = _count_bicolored_cycles(coloring, 4, adj)
                    coloring[v] = old_c
                    delta = new_cycles - cycles
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (v, new_c)
            if best_move is not None and best_delta < 0:
                v, new_c = best_move
                coloring[v] = new_c
                cycles += best_delta
                stall = 0
            else:
                stall += 1
                if stall > 30:
                    break
        if best4 is not None:
            break
    
    if best4 is not None and _is_valid(best4, 4, adj):
        return [int(c) for c in best4]
    
    # Fallback: greedy
    coloring = [0] * n
    for v in range(n):
        used = set(coloring[nb] for nb in adj[v] if nb < v)
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    k = max(coloring) + 1
    if _is_valid(coloring, k, adj):
        return [int(c) for c in coloring]
    
    return list(range(n))
# EVOLVE-BLOCK-END
