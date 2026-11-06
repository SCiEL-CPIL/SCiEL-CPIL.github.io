import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # Map edge to its 1-based index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_to_idx[(u, v)] = i + 1
    
    def get_adjacency(K):
        """Get adjacency sets for first K edges."""
        adj = [set() for _ in range(n)]
        for i in range(K):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_feasible(adj):
        """Quick feasibility: min degree >= 2 and connectivity."""
        for i in range(n):
            if len(adj[i]) < 2:
                return False
        visited = [False] * n
        visited[0] = True
        stack = [0]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return count == n
    
    def find_cycle_cover(K, adj):
        """
        Find a minimum-weight cycle cover using Hungarian algorithm.
        Returns the assignment (list of next vertices) or None.
        """
        INF = 10**9
        cost = np.full((n, n), INF, dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if j in adj[i]:
                    key = (min(i,j), max(i,j))
                    cost[i][j] = edge_to_idx.get(key, INF)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Check if all assigned costs are finite (valid edges)
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] >= INF:
                return None
        
        return col_ind  # col_ind[i] = next vertex after i
    
    def cycles_from_assignment(assignment):
        """Extract cycles from a permutation (assignment)."""
        visited = [False] * n
        cycles = []
        for i in range(n):
            if not visited[i]:
                cycle = []
                curr = i
                while not visited[curr]:
                    visited[curr] = True
                    cycle.append(curr)
                    curr = assignment[curr]
                cycles.append(cycle)
        return cycles
    
    def merge_cycles(cycles, adj, edge_to_idx):
        """
        Merge disjoint cycles into a single Hamiltonian cycle using 2-opt swaps.
        Returns the Hamiltonian cycle as a list, or None if not possible.
        """
        if len(cycles) == 1:
            return cycles[0]
        
        # Try to merge cycles pairwise
        # For each pair of cycles, try to find edges that can be swapped
        # to merge them while maintaining valid edges
        
        best_cycle = None
        best_score = float('inf')
        
        # Try all pairs of cycles
        for i in range(len(cycles)):
            for j in range(i+1, len(cycles)):
                c1 = cycles[i]
                c2 = cycles[j]
                
                # Try all pairs of edges in c1 and c2
                for a_idx in range(len(c1)):
                    a1 = c1[a_idx]
                    a2 = c1[(a_idx+1) % len(c1)]
                    
                    for b_idx in range(len(c2)):
                        b1 = c2[b_idx]
                        b2 = c2[(b_idx+1) % len(c2)]
                        
                        # Check if swap edges exist: a1-b1 and a2-b2 (or a1-b2 and a2-b1)
                        # Option 1: remove a1-a2 and b1-b2, add a1-b1 and a2-b2
                        key1 = (min(a1,b1), max(a1,b1))
                        key2 = (min(a2,b2), max(a2,b2))
                        
                        if key1 in edge_to_idx and key2 in edge_to_idx:
                            # Both new edges exist in the prefix
                            # Create merged cycle
                            # c1: ... a1, a2 ... c2: ... b1, b2 ...
                            # After swap: ... a1, b1, (reverse c2 from b2 to b1), b2, a2 ...
                            
                            # Extract segments
                            # c1 from a2 to a1 (going forward)
                            idx_a1 = c1.index(a1)
                            idx_a2 = c1.index(a2)
                            # c2 from b2 to b1 (going forward)
                            idx_b1 = c2.index(b1)
                            idx_b2 = c2.index(b2)
                            
                            # Build new cycle
                            new_cycle = []
                            # Start from a2, go through c1 to a1
                            pos = idx_a2
                            while True:
                                new_cycle.append(c1[pos])
                                if c1[pos] == a1:
                                    break
                                pos = (pos + 1) % len(c1)
                            
                            # Then go to b1, through c2 reversed from b1 to b2
                            new_cycle.append(b1)
                            pos = (idx_b1 + 1) % len(c2)
                            while True:
                                new_cycle.append(c2[pos])
                                if c2[pos] == b2:
                                    break
                                pos = (pos + 1) % len(c2)
                            
                            # Then go to a2 (already have a2 at start, so close)
                            # Actually we need to verify this forms a valid cycle
                            
                            # Check all consecutive edges exist
                            valid = True
                            for k in range(len(new_cycle)):
                                u = new_cycle[k]
                                v = new_cycle[(k+1) % len(new_cycle)]
                                key = (min(u,v), max(u,v))
                                if key not in edge_to_idx:
                                    valid = False
                                    break
                            
                            if valid and len(new_cycle) == len(c1) + len(c2):
                                # Compute score
                                score = 0
                                for k in range(len(new_cycle)):
                                    u = new_cycle[k]
                                    v = new_cycle[(k+1) % len(new_cycle)]
                                    key = (min(u,v), max(u,v))
                                    score = max(score, edge_to_idx[key])
                                
                                if score < best_score:
                                    best_score = score
                                    # Replace cycles i and j with merged
                                    other_cycles = [c for k, c in enumerate(cycles) if k != i and k != j]
                                    new_cycles = other_cycles + [new_cycle]
                                    best_cycle = new_cycles
        
        if best_cycle is not None:
            return merge_cycles(best_cycle, adj, edge_to_idx)
        
        return None
    
    def find_hamilton_cycle(K):
        """Find a Hamilton cycle in the graph formed by first K edges."""
        adj = get_adjacency(K)
        
        if not check_feasible(adj):
            return None
        
        # Try cycle cover approach
        assignment = find_cycle_cover(K, adj)
        if assignment is not None:
            cycles = cycles_from_assignment(assignment)
            
            if len(cycles) == 1:
                # Already a Hamilton cycle
                return cycles[0]
            
            # Try to merge cycles
            merged = merge_cycles(cycles, adj, edge_to_idx)
            if merged is not None:
                # merged should be a single cycle now
                if len(merged) == 1:
                    return merged[0]
        
        # Fallback: backtracking search
        adj_list = [list(s) for s in adj]
        path = [0]
        used = [False] * n
        used[0] = True
        
        def backtrack(current, depth):
            if depth == n:
                return 0 in adj[current]
            
            neighbors = adj_list[current]
            unvisited_neighbors = [nb for nb in neighbors if not used[nb]]
            
            if not unvisited_neighbors:
                return False
            
            # Warnsdorff heuristic
            unvisited_neighbors.sort(key=lambda x: len([y for y in adj_list[x] if not used[y]]))
            
            for nb in unvisited_neighbors:
                path.append(nb)
                used[nb] = True
                
                if backtrack(nb, depth + 1):
                    return True
                
                path.pop()
                used[nb] = False
            
            return False
        
        if backtrack(0, 1):
            return path
        return None
    
    # Binary search for minimal K
    low = n
    high = num_edges
    best_cycle = None
    best_K = num_edges + 1
    
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        cycle = find_hamilton_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1
    
    if best_cycle is None:
        # Try progressively larger K
        for K in range(n, num_edges + 1):
            if time.time() >= deadline:
                break
            cycle = find_hamilton_cycle(K)
            if cycle is not None:
                best_cycle = cycle
                best_K = K
                break
    
    if best_cycle is None:
        # Last resort
        best_cycle = list(range(n))
    
    return best_cycle

# EVOLVE-BLOCK-END
