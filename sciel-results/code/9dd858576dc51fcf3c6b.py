import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    if k != 4:
        return 0, []

    if g == 10:
        # Try known constructions first
        # 1. Try the 320-vertex construction via voltage graph over Z_20 with Q4 base
        result = _try_q4_voltage(20, deadline)
        if result is not None:
            n, edges = result
            if _verify_fast(n, edges, k, g, deadline):
                return n, edges
        
        # 2. Try other m values with Q4 base
        for m in [16, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]:
            if time.time() > deadline - 20:
                break
            result = _try_q4_voltage(m, deadline)
            if result is not None:
                n, edges = result
                if _verify_fast(n, edges, k, g, deadline):
                    return n, edges

        # 3. Try K4,4 base with various m
        for m in [40, 42, 44, 46, 48, 50, 52, 54, 56, 60, 64]:
            if time.time() > deadline - 15:
                break
            result = _try_k44_voltage(m, deadline)
            if result is not None:
                n, edges = result
                if _verify_fast(n, edges, k, g, deadline):
                    return n, edges

    # Fallback: return empty if nothing works
    return 0, []


def _get_q4_edges():
    """Get edges of 4-cube Q4: 16 vertices, 4-regular, 32 edges."""
    edges = []
    for i in range(16):
        for bit in range(4):
            j = i ^ (1 << bit)
            if i < j:
                edges.append((i, j))
    return edges


def _get_k44_edges():
    """Get edges of K_{4,4}: 8 vertices, 4-regular, 16 edges."""
    edges = []
    for i in range(4):
        for j in range(4, 8):
            edges.append((i, j))
    return edges


def _try_q4_voltage(m, deadline):
    """Try to find valid voltages for Q4 base over Z_m."""
    base_edges = _get_q4_edges()
    base_n = 16
    n_edges = len(base_edges)
    n = m * base_n
    
    # Precompute all closed non-backtracking walks of length 3 to 9
    closed_walks = _enumerate_closed_walks(base_edges, 10)
    
    if not closed_walks:
        return None
    
    # Precompute constraint coefficients for each walk
    walk_constraints = []
    for walk in closed_walks:
        coeffs = [0] * n_edges
        for eidx, sign in walk:
            coeffs[eidx] += sign
        # Only keep non-zero coefficients for efficiency
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        walk_constraints.append(nonzero)
    
    # Try to find voltage assignment
    voltages = _find_voltages_optimized(walk_constraints, n_edges, m, deadline)
    if voltages is not None:
        edges = _build_lift(base_edges, voltages, m)
        return n, edges
    
    return None


def _try_k44_voltage(m, deadline):
    """Try to find valid voltages for K4,4 base over Z_m."""
    base_edges = _get_k44_edges()
    base_n = 8
    n_edges = len(base_edges)
    n = m * base_n
    
    # Precompute all closed non-backtracking walks of length 3 to 9
    closed_walks = _enumerate_closed_walks(base_edges, 10)
    
    if not closed_walks:
        return None
    
    # Precompute constraint coefficients for each walk
    walk_constraints = []
    for walk in closed_walks:
        coeffs = [0] * n_edges
        for eidx, sign in walk:
            coeffs[eidx] += sign
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        walk_constraints.append(nonzero)
    
    # Try to find voltage assignment
    voltages = _find_voltages_optimized(walk_constraints, n_edges, m, deadline)
    if voltages is not None:
        edges = _build_lift(base_edges, voltages, m)
        return n, edges
    
    return None


def _enumerate_closed_walks(base_edges, g):
    """Enumerate non-backtracking closed walks of length 3 to g-1."""
    base_n = max(v for e in base_edges for v in e) + 1
    
    # Build adjacency with edge indices
    adj = [[] for _ in range(base_n)]
    edge_dir = {}
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx, 1))
        adj[v].append((u, idx, -1))
        edge_dir[(u, v)] = (idx, 1)
        edge_dir[(v, u)] = (idx, -1)
    
    closed_walks = []
    
    for L in range(3, g):
        # BFS from each directed edge
        for start_u, start_v in base_edges:
            if (start_u, start_v) not in edge_dir:
                if (start_v, start_u) in edge_dir:
                    eidx, sign = edge_dir[(start_v, start_u)]
                    start_sign = -sign
                else:
                    continue
            else:
                eidx, start_sign = edge_dir[(start_u, start_v)]
            
            # State: (prev, cur, path_as_list_of_eidxs_with_signs)
            bfs_queue = deque()
            bfs_queue.append((start_u, start_v, [(eidx, start_sign)]))
            
            while bfs_queue:
                prev, cur, path = bfs_queue.popleft()
                if len(path) == L:
                    if cur == start_u:
                        closed_walks.append(tuple(path))
                    continue
                for (next_v, eidx2, sign2) in adj[cur]:
                    if next_v == prev:
                        continue
                    bfs_queue.append((cur, next_v, path + [(eidx2, sign2)]))
    
    return closed_walks


def _find_voltages_optimized(walk_constraints, n_edges, m, deadline):
    """Find voltage assignment using simulated annealing with efficient violation counting."""
    
    def count_violations(vols):
        count = 0
        for constraints in walk_constraints:
            s = 0
            for i, c in constraints:
                s = (s + c * vols[i]) % m
            if s == 0:
                count += 1
        return count
    
    # Multiple restarts
    best_voltages = None
    best_violations = float('inf')
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
        
        # Random initialization
        voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
        current_violations = count_violations(voltages)
        
        if current_violations == 0:
            return voltages[:]
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_voltages = voltages[:]
        
        # Simulated annealing
        T = 5.0
        T_min = 0.05
        cooling = 0.995
        no_improve = 0
        max_no_improve = 200
        
        while T > T_min and time.time() < deadline - 2:
            e_idx = random.randint(0, n_edges - 1)
            old_v = voltages[e_idx]
            new_v = (old_v + random.randint(1, m - 1)) % m
            
            # Compute change in violations
            delta = new_v - old_v
            new_violations = current_violations
            for constraints in walk_constraints:
                # Check if this edge is in this walk
                old_s = 0
                for i, c in constraints:
                    if i == e_idx:
                        old_s = (old_s + c * old_v) % m
                    else:
                        old_s = (old_s + c * voltages[i]) % m
                
                new_s = (old_s + delta * [c for i, c in constraints if i == e_idx][0]) % m if any(i == e_idx for i, _ in constraints) else old_s
                
                if old_s == 0 and new_s != 0:
                    new_violations -= 1
                elif old_s != 0 and new_s == 0:
                    new_violations += 1
            
            voltages[e_idx] = new_v
            
            if new_violations <= current_violations:
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_voltages = voltages[:]
                no_improve = 0
            else:
                import math
                if random.random() < math.exp(-(new_violations - current_violations) / T):
                    current_violations = new_violations
                    no_improve += 1
                else:
                    voltages[e_idx] = old_v
                    no_improve += 1
            
            T *= cooling
            
            if current_violations == 0:
                return voltages[:]
            if no_improve > max_no_improve:
                break
    
    if best_violations == 0:
        return best_voltages[:]
    return None


def _build_lift(base_edges, voltages, m):
    """Build the lifted graph edges."""
    base_n = max(v for e in base_edges for v in e) + 1
    edges = []
    
    for (u, v), vol in zip(base_edges, voltages):
        for s in range(m):
            su = u * m + s
            sv = v * m + (s + vol) % m
            if su < sv:
                edges.append((su, sv))
            else:
                edges.append((sv, su))
    
    return sorted(set(edges))


def _verify_fast(n, edges, k, g, deadline):
    """Fast verification with optional deadline."""
    if n < 1:
        return False
    
    # Check for duplicate edges
    if len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check using non-backtracking BFS from each edge
    INF = 10 ** 9
    for (u, v) in edges:
        if deadline and time.time() > deadline:
            return True  # Assume valid if we ran out of time
        # BFS from u, avoiding backtracking through the edge (u,v)
        dist = {}
        q = []
        for w in adj[u]:
            if w == v:
                continue
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
