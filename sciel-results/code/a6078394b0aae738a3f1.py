"""Improved program: circulant + backtracking + simulated annealing for max edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import sys
import math
import random
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_circulant_cycles(n, all_edges, used_edges, deadline):
    """Find all valid circulant Hamiltonian cycles with unused edges."""
    cycles = []
    for k in range(1, n):
        if time.time() > deadline:
            break
        if _gcd(k, n) != 1:
            continue
        
        valid = True
        for i in range(n):
            u = i
            v = (i + k) % n
            ek = _edge_key(u, v)
            if ek not in all_edges or ek in used_edges:
                valid = False
                break
        
        if valid:
            cycle = []
            v = 0
            for _ in range(n):
                cycle.append(v)
                v = (v + k) % n
            cycles.append(cycle)
    
    return cycles

def _find_hamiltonian_cycle_bt(adj, n, used_edges, deadline, start_vertex=0):
    """Find a Hamiltonian cycle using backtracking with aggressive pruning."""
    
    avail = [set() for _ in range(n)]
    for v in range(n):
        for nb in adj[v]:
            if _edge_key(v, nb) not in used_edges:
                avail[v].add(nb)
    
    deg = [len(avail[v]) for v in range(n)]
    neighbor_order = {v: sorted(avail[v], key=lambda x: (deg[x], x)) for v in range(n)}
    
    def _check_connectivity(avail_sets, visited_set, current):
        unvisited = [v for v in range(n) if v not in visited_set]
        if not unvisited:
            return True
        
        visited_bfs = set()
        queue = deque([current])
        visited_bfs.add(current)
        
        while queue:
            node = queue.popleft()
            for nb in avail_sets[node]:
                if nb not in visited_bfs and (nb not in visited_set or nb == current):
                    visited_bfs.add(nb)
                    queue.append(nb)
        
        for v in unvisited:
            if v not in visited_bfs:
                return False
        return True
    
    def backtrack(visited, path, current, depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            if path[0] in avail[current]:
                return path[:]
            return None
        
        if not _check_connectivity(avail, visited, current):
            return None
        
        candidates = []
        for neighbor in neighbor_order[current]:
            if neighbor not in visited:
                candidates.append(neighbor)
        
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            result = backtrack(visited, path, neighbor, depth + 1)
            if result is not None:
                return result
            path.pop()
            visited.discard(neighbor)
        return None
    
    visited = {start_vertex}
    path = [start_vertex]
    result = backtrack(visited, path, start_vertex, 1)
    if result is not None:
        return result
    
    for start in range(n):
        if time.time() > deadline:
            break
        visited = {start}
        path = [start]
        result = backtrack(visited, path, start, 1)
        if result is not None:
            return result
    
    return None

def _find_hamiltonian_cycle_sa(adj, n, used_edges, deadline, max_restarts=50):
    """
    Find a Hamiltonian cycle using simulated annealing with 2-opt moves.
    This explores the space of permutations and repairs invalid transitions.
    """
    # Build adjacency set for quick edge lookup
    edge_set = set()
    for u, v in P['edges']:
        ek = _edge_key(u, v)
        if ek not in used_edges:
            edge_set.add(ek)
    
    def count_invalid(permutation):
        """Count the number of consecutive pairs (including wrap) that are NOT edges in remaining graph."""
        invalid = 0
        for i in range(n):
            u = permutation[i]
            v = permutation[(i + 1) % n]
            if _edge_key(u, v) not in edge_set:
                invalid += 1
        return invalid
    
    def two_opt_swap(permutation, i, j):
        """Perform a 2-opt move: reverse the segment between positions i and j."""
        new_perm = permutation[:i+1] + permutation[i+1:j+1][::-1] + permutation[j+1:]
        return new_perm
    
    best_cycle = None
    best_invalid = n + 1
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start with a random permutation
        perm = list(range(n))
        random.shuffle(perm)
        
        current_invalid = count_invalid(perm)
        best_invalid_local = current_invalid
        best_perm_local = perm[:]
        
        if current_invalid == 0:
            return perm
        
        temperature = 1.0
        cooling = 0.995
        no_improve = 0
        max_no_improve = 2000
        
        while temperature > 0.01 and no_improve < max_no_improve:
            if time.time() > deadline:
                break
            
            # Random 2-opt move
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if i == j or i == j - 1:
                no_improve += 1
                continue
            
            new_perm = two_opt_swap(perm, i, j)
            new_invalid = count_invalid(new_perm)
            
            delta = new_invalid - current_invalid
            
            # Accept if improvement or with probability exp(-delta/temperature)
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                perm = new_perm
                current_invalid = new_invalid
                no_improve = 0
                if current_invalid < best_invalid_local:
                    best_invalid_local = current_invalid
                    best_perm_local = perm[:]
                    if current_invalid == 0:
                        return perm
            else:
                no_improve += 1
            
            temperature *= cooling
        
        if best_invalid_local < best_invalid:
            best_invalid = best_invalid_local
            best_cycle = best_perm_local
    
    if best_cycle is not None and count_invalid(best_cycle) == 0:
        return best_cycle
    
    return None

def _find_hamiltonian_cycle(adj, n, used_edges, deadline, start_vertex=0):
    """
    Combined approach: try backtracking first (fast if it works),
    then fall back to simulated annealing.
    """
    # Try backtracking with a short time budget
    bt_deadline = min(deadline, time.time() + 3.0)
    result = _find_hamiltonian_cycle_bt(adj, n, used_edges, bt_deadline, start_vertex)
    if result is not None:
        return result
    
    # Try simulated annealing with remaining time
    sa_deadline = deadline
    result = _find_hamiltonian_cycle_sa(adj, n, used_edges, sa_deadline)
    if result is not None:
        return result
    
    # Try backtracking with more time on different starts
    for start in range(n):
        if time.time() > deadline:
            break
        bt_deadline2 = min(deadline, time.time() + 2.0)
        result = _find_hamiltonian_cycle_bt(adj, n, used_edges, bt_deadline2, start)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Leave 8s safety margin before 90s kill
    
    adj = _build_adj()
    n = P['n']
    
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    used_edges = set()
    cycles = []
    
    # Step 1: Find all circulant cycles (fast, deterministic)
    circulant_cycles = _find_circulant_cycles(n, all_edges, used_edges, deadline)
    circulant_cycles.sort(key=lambda c: abs(c[1] - c[0]) if len(c) > 1 else 0)
    
    for cycle in circulant_cycles:
        if time.time() > deadline - 5:
            break
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = _edge_key(u, v)
            if edge in used_edges:
                valid = False
                break
            used_edges.add(edge)
        if valid:
            cycles.append(cycle)
    
    # Step 2: Use combined backtracking + SA to find additional Hamiltonian cycles
    max_cycles = 3  # Theoretical max for 6-regular graph
    while len(cycles) < max_cycles:
        if time.time() > deadline - 5:
            break
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
        
        cycle = None
        # Try multiple starting vertices with time allocation
        time_per_start = max(1.0, remaining_time / (n + 5))
        
        for start in range(n):
            if time.time() > deadline - 5:
                break
            sub_deadline = min(deadline - 5, time.time() + time_per_start)
            cycle = _find_hamiltonian_cycle(adj, n, used_edges, sub_deadline, start)
            if cycle is not None:
                break
        
        if cycle is None:
            # Try SA with more time on a random start
            if time.time() < deadline - 3:
                sa_deadline = min(deadline - 3, time.time() + 10)
                cycle = _find_hamiltonian_cycle_sa(adj, n, used_edges, sa_deadline, max_restarts=100)
        
        if cycle is None:
            break
        
        # Verify and add
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = _edge_key(u, v)
            if edge in used_edges:
                valid = False
                break
        if valid:
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                edge = _edge_key(u, v)
                used_edges.add(edge)
            cycles.append(cycle)
        else:
            break
    
    # Ensure we have at least one cycle (fallback)
    if not cycles and time.time() < deadline:
        cycle = _find_hamiltonian_cycle(adj, n, used_edges, deadline, 0)
        if cycle is not None:
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                edge = _edge_key(u, v)
                used_edges.add(edge)
            cycles.append(cycle)
    
    # Cap at 10 cycles (though max is 3 for this graph)
    cycles = cycles[:10]
    
    # Convert to lists of Python ints
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
