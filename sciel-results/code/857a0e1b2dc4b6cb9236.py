"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency with edge IDs
    adj = [[] for _ in range(n)]
    edge_u = [e[0] for e in edges]
    edge_v = [e[1] for e in edges]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    deadline = time.time() + 50
    
    # Structural change: Use a randomized edge-partitioning approach.
    # Assign each edge to one of K color classes (K=3). Then check if each
    # color class forms a Hamiltonian cycle. Use simulated annealing to
    # optimize the coloring.
    
    K = 3  # target number of cycles
    best_cycles = None
    best_score = -1
    
    # Precompute: for a given edge subset (represented as a set of edge indices),
    # check if it forms a single Hamiltonian cycle on all n vertices.
    def check_hamiltonian_cycle(edge_indices):
        """Check if the given edge indices form exactly one Hamiltonian cycle."""
        if len(edge_indices) != n:
            return None
        # Build adjacency for just these edges
        local_adj = [[] for _ in range(n)]
        for eid in edge_indices:
            u, v = edge_u[eid], edge_v[eid]
            local_adj[u].append(v)
            local_adj[v].append(u)
        # Check each vertex has degree 2
        for i in range(n):
            if len(local_adj[i]) != 2:
                return None
        # Check it's a single cycle (not multiple cycles)
        visited = [False] * n
        visited[0] = True
        count = 1
        cur = 0
        prev = -1
        for _ in range(n):
            nxts = local_adj[cur]
            nxt = nxts[0] if nxts[0] != prev else nxts[1]
            if nxt == 0:
                break
            if visited[nxt]:
                return None
            visited[nxt] = True
            count += 1
            prev, cur = cur, nxt
        if count != n:
            return None
        # Reconstruct the cycle
        visited2 = [False] * n
        visited2[0] = True
        cycle = [0]
        cur = 0
        prev = -1
        for _ in range(n - 1):
            nxts = local_adj[cur]
            nxt = nxts[0] if nxts[0] != prev else nxts[1]
            visited2[nxt] = True
            cycle.append(nxt)
            prev, cur = cur, nxt
        return cycle
    
    # Fast initial heuristic: try to find 3 edge-disjoint Hamiltonian cycles using backtracking
    # with constraint propagation (degree checks)
    
    def find_ham_cycles_backtrack(used_edges, num_cycles_target, start_vertex=0):
        """Find num_cycles_target edge-disjoint Hamiltonian cycles using backtracking."""
        cycles = []
        
        def find_one_cycle(used_e, start):
            """Find one Hamiltonian cycle starting from start, avoiding used_e edges."""
            visited = [False] * n
            visited[start] = True
            path = [start]
            path_edges = []
            
            def dfs(current, depth):
                if time.time() > deadline:
                    return None
                if depth == n - 1:
                    for (nb, eid) in adj[current]:
                        if nb == start and eid not in used_e:
                            return (path[:], path_edges[:])
                    return None
                # Try neighbors in order of fewest unused edges first (heuristic)
                candidates = []
                for (nb, eid) in adj[current]:
                    if nb != start and not visited[nb] and eid not in used_e:
                        candidates.append((nb, eid))
                # Sort by degree in remaining graph (fewer options first)
                candidates.sort(key=lambda x: len([e for e in adj[x[0]] if e[1] not in used_e]))
                for (nb, eid) in candidates:
                    visited[nb] = True
                    path.append(nb)
                    path_edges.append(eid)
                    result = dfs(nb, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    path_edges.pop()
                    visited[nb] = False
                return None
            
            return dfs(start, 0)
        
        for _ in range(num_cycles_target):
            if time.time() > deadline:
                break
            found = None
            for start in range(n):
                if time.time() > deadline:
                    break
                result = find_one_cycle(used_edges, start)
                if result is not None:
                    found = result
                    break
            if found is None:
                break
            cycle_path, cycle_eids = found
            cycles.append(cycle_path)
            used_edges = set(used_edges) | set(cycle_eids)
        
        return cycles
    
    # Try to find 3 cycles quickly with backtracking
    best_cycles = find_ham_cycles_backtrack(set(), 3)
    best_score = len(best_cycles)
    
    # If we found at least 1 cycle, try to improve with simulated annealing
    if best_score < 3 and time.time() < deadline - 10:
        # Simulated annealing on edge coloring
        def evaluate_coloring(colors):
            """Return (num_valid_cycles, list_of_cycles)."""
            groups = [[] for _ in range(K)]
            for i in range(m):
                groups[colors[i]].append(i)
            valid = 0
            cycles = []
            for k in range(K):
                if len(groups[k]) != n:
                    continue
                cyc = check_hamiltonian_cycle(groups[k])
                if cyc is not None:
                    valid += 1
                    cycles.append(cyc)
            return valid, cycles
        
        # Initial random coloring
        colors = [random.randint(0, K - 1) for _ in range(m)]
        current_valid, current_cycles = evaluate_coloring(colors)
        if current_valid > best_score:
            best_score = current_valid
            best_cycles = list(current_cycles)
        
        T = 2.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < deadline - 5:
            # Pick a random edge and reassign its color
            i = random.randint(0, m - 1)
            old_color = colors[i]
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                continue
            colors[i] = new_color
            
            new_valid, new_cycles = evaluate_coloring(colors)
            delta = new_valid - current_valid
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_valid = new_valid
                current_cycles = new_cycles
                if current_valid > best_score:
                    best_score = current_valid
                    best_cycles = list(new_cycles)
            else:
                colors[i] = old_color
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Verify and return best cycles found
    if best_cycles and len(best_cycles) > 0:
        result = []
        used_edges = set()
        for cyc in best_cycles:
            cyc_edges = set()
            valid_cycle = True
            for j in range(len(cyc)):
                u = cyc[j]
                v = cyc[(j + 1) % len(cyc)]
                # Find edge ID
                found = False
                for eid in range(m):
                    if (edge_u[eid] == u and edge_v[eid] == v) or (edge_u[eid] == v and edge_v[eid] == u):
                        if eid not in used_edges:
                            cyc_edges.add(eid)
                            found = True
                            break
                        else:
                            found = False
                            break
                if not found:
                    valid_cycle = False
                    break
            if not valid_cycle or len(cyc_edges) != len(cyc):
                continue
            used_edges.update(cyc_edges)
            result.append(list(cyc))
        if result:
            return result
    
    # Final fallback: try to find at least 1 cycle with backtracking
    fallback_cycles = find_ham_cycles_backtrack(set(), 1)
    if fallback_cycles:
        return fallback_cycles
    
    return [list(range(n))]

# EVOLVE-BLOCK-END
