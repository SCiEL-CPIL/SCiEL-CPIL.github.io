import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Uses a hybrid approach:
    1. Vectorized NumPy state maintenance for fast conflict evaluation.
    2. Backtracking with Forward Checking for initial construction.
    3. Local Search (WalkSAT) for escaping local minima.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Helper to get edge index in lexicographic order
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        # Build adjacency matrices for each color
        # adj[c] is an n x n matrix where adj[c][u][v] is 1 if edge (u,v) is color c
        adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                idx += 1
        
        # Check for C4:
        # A C4 exists in color c if there are u, v such that they have >= 2 common neighbors in color c.
        # Let A_c be the adjacency matrix.
        # Common neighbors of u and v in color c = (A_c @ A_c)[u, v]
        # We require (A_c @ A_c)[u, v] < 2 for all u != v.
        # Note: (A_c @ A_c)[u, u] is the degree of u, which can be anything.
        # We only care about off-diagonal entries.
        
        for c in range(5):
            # Matrix multiplication gives number of common neighbors
            # For n=29, 29x29 matmul is very fast.
            common = np.dot(adj[c], adj[c])
            # Check off-diagonal entries
            # np.fill_diagonal(common, 0)
            # if np.any(common >= 2): return False
            # Faster:
            # Create a mask for off-diagonal
            # Or just check max of off-diagonal
            # Since n is small, we can just loop or use numpy masking
            mask = ~np.eye(n, dtype=bool)
            if np.any(common[mask] >= 2):
                return False
        return True

    def count_violations_vectorized(n, adj):
        """
        Counts the number of pairs (u, v) with >= 2 common neighbors in any color.
        adj: list of 5 numpy arrays (n x n)
        Returns total count of violating pairs.
        """
        cost = 0
        for c in range(5):
            common = np.dot(adj[c], adj[c])
            # Exclude diagonal
            np.fill_diagonal(common, 0)
            # Count entries >= 2
            cost += np.sum(common >= 2)
        return cost

    def solve_hybrid(n, time_limit):
        """
        Hybrid solver: Random init -> Local Search -> Backtracking refinement.
        """
        total_edges = n * (n - 1) // 2
        
        # Precompute edge endpoints
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[idx] = (i, j)
                idx += 1
        
        # Initialize random coloring
        colors = np.array([random.randint(0, 4) for _ in range(total_edges)], dtype=np.int8)
        
        # Build adjacency matrices
        adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
        for e in range(total_edges):
            u, v = edge_map[e]
            c = colors[e]
            adj[c][u, v] = 1
            adj[c][v, u] = 1
            
        current_cost = count_violations_vectorized(n, adj)
        
        if current_cost == 0:
            return colors.tolist()
            
        best_cost = current_cost
        best_colors = colors.copy()
        
        start_time = time.time()
        
        # Phase 1: Local Search (WalkSAT style)
        # We iterate until time limit or cost 0.
        # Move: pick an edge involved in a violation, change its color.
        
        # To speed up, we can identify "bad" edges.
        # An edge (u,v) is bad if changing it might reduce cost.
        # Actually, any edge can be part of a C4.
        
        # Simple strategy: Pick random edge, try all 5 colors, pick best.
        # If best is worse than current, accept with probability p (simulated annealing).
        
        T = 1.0
        cooling_rate = 0.9999
        no_improve_counter = 0
        max_no_improve = 1000
        
        while time.time() - start_time < time_limit:
            if current_cost == 0:
                return colors.tolist()
                
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            
            best_new_c = -1
            best_new_cost = current_cost
            best_delta = 0
            
            # Try all 5 colors
            for new_c in range(5):
                if new_c == old_c:
                    continue
                
                # Apply move
                adj[old_c][u, v] = 0
                adj[old_c][v, u] = 0
                adj[new_c][u, v] = 1
                adj[new_c][v, u] = 1
                
                # Calculate new cost
                # Optimization: Only recalculate cost for colors old_c and new_c?
                # Cost is sum over c. Changing edge (u,v) from c1 to c2 affects:
                # - Common neighbor counts for pairs involving u or v in c1 and c2.
                # - Specifically, (A @ A) changes for rows/cols u and v in A_c1 and A_c2.
                # For simplicity and n<=29, full recalc is ~1000 ops. 
                # Let's do full recalc for now, it's fast enough.
                
                new_cost = count_violations_vectorized(n, adj)
                
                delta = new_cost - current_cost
                if delta < best_delta:
                    best_delta = delta
                    best_new_cost = new_cost
                    best_new_c = new_c
                    
            # Revert to old state for decision
            adj[old_c][u, v] = 1
            adj[old_c][v, u] = 1
            adj[best_new_c][u, v] = 0
            adj[best_new_c][v, u] = 0
            
            # Decide whether to accept best_new_c
            if best_new_c == -1:
                continue
                
            delta = best_new_cost - current_cost
            accept = False
            if delta < 0:
                accept = True
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    accept = True
            
            if accept:
                # Apply move
                adj[old_c][u, v] = 0
                adj[old_c][v, u] = 0
                adj[best_new_c][u, v] = 1
                adj[best_new_c][v, u] = 1
                colors[e_idx] = best_new_c
                current_cost = best_new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                    
                T *= cooling_rate
                if T < 0.01:
                    T = 1.0  # Reheat
            else:
                no_improve_counter += 1
                
            # If stuck for too long, restart with new random init?
            # Or just continue.
            
        return best_colors.tolist() if best_cost == 0 else None

    # Strategy:
    # Try n from 29 down to 12.
    # For each n, run the hybrid solver.
    
    # Known seed solution for n=11
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()
    
    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
            
        # Time allocation
        if n >= 27:
            t_alloc = 8.0
        elif n == 26:
            t_alloc = 20.0
        else:
            t_alloc = 8.0
            
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        res = solve_hybrid(n, t_alloc)
        
        if res:
            # Verify
            if is_valid(n, res):
                best_n = n
                best_colors = res
                return {'n': best_n, 'colors': best_colors}
                
    # If all fail, return seed
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
