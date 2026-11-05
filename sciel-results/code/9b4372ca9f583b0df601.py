import time
import random
import sys
import numpy as np
from collections import defaultdict

# Check for scipy availability
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csc_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _enumerate_cycles(adj, n, max_len=12, max_cycles=5000):
    """
    Enumerate all simple cycles up to max_len.
    To avoid duplicates, we only count cycles where the smallest vertex in the cycle is the start vertex.
    """
    cycles = []
    # Map edge to index for quick lookup? Not strictly needed for enumeration, but good for later.
    
    # For each vertex, do a DFS to find cycles
    # Optimization: Only start DFS from vertex v if v is the smallest vertex in the potential cycle.
    # This is hard to enforce directly during DFS without knowing the cycle.
    # Instead, we can generate all cycles and deduplicate using a canonical form (frozenset of edges).
    
    seen_cycles = set()
    
    def dfs(start, current, path, visited):
        if len(cycles) >= max_cycles:
            return
        
        if len(path) > 1 and current == start and len(path) >= 3:
            # Found a cycle
            # Canonical representation: frozenset of edges
            cycle_edges = set()
            for i in range(len(path)):
                u = path[i]
                v = path[(i+1) % len(path)]
                if u > v: u, v = v, u
                cycle_edges.add((u, v))
            key = frozenset(cycle_edges)
            if key not in seen_cycles:
                seen_cycles.add(key)
                # Store the path in canonical order starting with min vertex
                min_idx = path.index(min(path))
                canon_path = path[min_idx:] + path[:min_idx]
                cycles.append(canon_path)
            return
        
        if len(path) >= max_len:
            return
            
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 2:
                # Close cycle
                dfs(start, start, path + [start], visited)
            elif neighbor not in visited:
                visited.add(neighbor)
                dfs(start, neighbor, path + [neighbor], visited)
                visited.remove(neighbor)
                
    for v in range(n):
        if len(cycles) >= max_cycles:
            break
        # To reduce duplicates, we can restrict starting vertices?
        # Actually, the canonical form check handles duplicates.
        # But we can speed up by only starting from v if v is smaller than all other vertices in the cycle?
        # That's not known in advance.
        # However, we can prune: if we are at 'current' and 'neighbor' < 'start', we can skip?
        # No, because the cycle might go through 'start' later? No, start is fixed.
        # A common trick: only allow vertices > start in the path (except closing to start).
        # This ensures start is the minimum vertex in the cycle.
        
        def dfs_min(start, current, path, visited):
            if len(cycles) >= max_cycles:
                return
            
            if current == start and len(path) >= 3:
                # Since we only visited vertices > start, start is the min.
                # Path is already canonical (starts with min).
                cycle_edges = set()
                for i in range(len(path)):
                    u = path[i]
                    v = path[(i+1) % len(path)]
                    if u > v: u, v = v, u
                    cycle_edges.add((u, v))
                key = frozenset(cycle_edges)
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycles.append(path[:])
                return
            
            if len(path) >= max_len:
                return
                
            for neighbor in adj[current]:
                if neighbor == start:
                    if len(path) >= 2: # Need at least 2 edges to close, i.e., len(path)>=3
                        dfs_min(start, start, path + [start], visited)
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    dfs_min(start, neighbor, path + [neighbor], visited)
                    visited.remove(neighbor)
                    
        visited = {v}
        dfs_min(v, v, [v], visited)
        
    return cycles

def _solve_ilp(edges, n, deadline):
    """
    Solve the decomposition using ILP.
    """
    if not HAS_SCIPY:
        return None
        
    adj = [set() for _ in range(n)]
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        adj[u].add(v)
        adj[v].add(u)
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i # Just in case, though we use sorted
        
    m = len(edges)
    
    # Enumerate cycles
    # Limit max_len to keep enumeration fast. 
    # For n=20-30, max_len=12 is usually enough to find most useful cycles.
    # If the optimal solution uses a very long cycle, we might miss it, but ILP on short cycles is a good baseline.
    cycles = _enumerate_cycles(adj, n, max_len=10, max_cycles=2000)
    
    # Variables:
    # x[0..C-1] for cycles
    # x[C..C+m-1] for edges
    C = len(cycles)
    total_vars = C + m
    
    # Objective: minimize sum(x)
    c = np.ones(total_vars)
    
    # Constraints: For each edge, sum of x_vars covering it == 1
    # A matrix of shape (m, total_vars)
    A = np.zeros((m, total_vars))
    
    for i, (u, v) in enumerate(edges):
        # Edge variable
        A[i, C + i] = 1.0
        
    for k, cyc in enumerate(cycles):
        for j in range(len(cyc)):
            u = cyc[j]
            v = cyc[(j+1) % len(cyc)]
            # Find edge index
            key = (u, v) if u < v else (v, u)
            if key in edge_to_idx:
                idx = edge_to_idx[key]
                A[idx, k] = 1.0
            else:
                # Should not happen if enumeration is correct
                pass
                
    # Use sparse matrix for efficiency
    A_sparse = csc_matrix(A)
    
    constraints = LinearConstraint(A_sparse, lb=np.ones(m), ub=np.ones(m))
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(total_vars))
        
        if res.status == 0: # Optimal
            x = res.x
            # Extract solution
            pieces = []
            for k in range(C):
                if x[k] > 0.5:
                    pieces.append(cycles[k])
            for i in range(m):
                if x[C + i] > 0.5:
                    u, v = edges[i]
                    pieces.append([u, v])
            return pieces
        else:
            return None
    except Exception:
        return None

def _solve_greedy(edges, n, deadline):
    """
    Fallback greedy solver.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a vertex with max degree
        start = -1
        max_deg = 0
        for v in range(n):
            if len(adj[v]) > max_deg:
                max_deg = len(adj[v])
                start = v
        if start == -1:
            break
            
        # Try to find a cycle starting from start
        # Simple DFS to find a cycle
        found_cycle = None
        # Try to find a cycle of length >= 3
        # We can use a randomized DFS with backtracking limited by time
        
        # For speed, we just try a few random walks
        for _ in range(10):
            path = [start]
            visited = {start}
            current = start
            # Randomize neighbor order
            nbrs = list(adj[current])
            random.shuffle(nbrs)
            
            while True:
                nexts = [nb for nb in nbrs if nb not in visited]
                if not nexts:
                    break
                # Check if we can close
                if start in adj[current] and len(path) >= 3:
                    found_cycle = path
                    break
                # Pick next
                # Prefer lower degree?
                nexts.sort(key=lambda x: len(adj[x]))
                next_v = nexts[0]
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                nbrs = list(adj[current])
                random.shuffle(nbrs)
                
            if found_cycle:
                break
                
        if found_cycle and len(found_cycle) >= 3:
            # Remove edges
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i+1) % len(found_cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
            remaining -= len(found_cycle)
            pieces.append(found_cycle)
        else:
            # No cycle, remove an edge
            if not adj[start]:
                continue
            nbrs = list(adj[start])
            nbrs.sort(key=lambda x: len(adj[x]))
            b = nbrs[0]
            adj[start].discard(b)
            adj[b].discard(start)
            remaining -= 1
            pieces.append([start, b])
            
    # Leftover
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])
                
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # Try ILP first
    # Give it a chunk of time
    ilp_deadline = time.time() + (deadline - time.time()) * 0.7
    if time.time() < ilp_deadline:
        res = _solve_ilp(edges, n, ilp_deadline)
        if res:
            best = res
            best_count = len(res)
            
    # If ILP failed or we have time, try greedy
    if time.time() < deadline:
        greedy_deadline = deadline - 0.5
        res_g = _solve_greedy(edges, n, greedy_deadline)
        if res_g and len(res_g) < best_count:
            best = res_g
            best_count = len(res_g)
            
    if best is None:
        # Absolute fallback: just edges
        best = [list(e) for e in edges]
        
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Calculate deadline for this graph
        # Ensure we don't exceed total budget
        current_elapsed = time.time() - start_time
        remaining_total = total_budget - current_elapsed - 2.0 # 2s safety margin
        
        # Allocate remaining time to remaining graphs
        graphs_left = len(specs) - i
        allocated = remaining_total / graphs_left
        
        graph_deadline = time.time() + allocated
        
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total pieces: {total}")
