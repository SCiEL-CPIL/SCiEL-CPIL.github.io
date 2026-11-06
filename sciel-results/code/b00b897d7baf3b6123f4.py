import time
import random
import sys

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0  # Safety margin for 120s limit

    def has_path_len_5(adj, u, v):
        """
        Check if there is a path of length 5 from u to v in the graph represented by adj (list of bitmasks).
        adj[i] is a bitmask of neighbors of i.
        """
        # We want to find a sequence u - a - b - c - d - v
        # This is equivalent to finding a path of length 4 from u to any neighbor of v,
        # or more simply, doing a BFS/DFS up to depth 5 from u and seeing if we hit v.
        
        # Optimization: We can do a bidirectional search or just a limited DFS.
        # Given n <= 19, the graph is small.
        
        # Let's use a simple DFS with pruning.
        # State: (current_node, depth, visited_mask)
        # We stop if depth == 5. If current_node == v at depth 5, return True.
        
        # To make it faster, we can check neighbors of v first.
        # If there is a path u->...->x->v of length 5, then x must be a neighbor of v.
        # So we check if there is a path of length 4 from u to any x in N(v).
        
        target_neighbors = adj[v]
        
        def dfs(current, depth, visited):
            if depth == 4:
                # Check if current is a neighbor of v
                if (adj[v] >> current) & 1:
                    return True
                return False
            
            if current == v:
                # We can't use v as an intermediate node in a simple cycle C6 
                # unless we are closing it, but the path definition u-a-b-c-d-v 
                # implies distinct vertices for a simple cycle.
                # However, the problem says "exact search for not necessarily induced cycles".
                # A C6 is a simple cycle. Vertices must be distinct.
                # So if we hit v before depth 5, it's not a valid C6 path (it would be a shorter cycle).
                # Wait, if we hit v at depth 3, that's a C4. If depth 2, C3.
                # We specifically need length 5.
                return False

            neighbors = adj[current] & ~visited
            # Iterate through neighbors
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False

        return dfs(u, 0, 1 << u)

    def solve(n, time_budget):
        """
        Greedy backtracking solver for n vertices.
        Returns a list of colors for edges (i,j) i<j or None.
        """
        deadline = time.time() + time_budget
        adj = [[0] * n for _ in range(4)]
        # colors[i][j] = color of edge (i,j), -1 if unassigned
        colors = [[-1] * n for _ in range(n)]
        
        # Order of edges to assign
        # Standard lexicographic order
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        edge_idx = 0
        num_edges = len(edges)
        
        # Recursion depth is num_edges
        # Python recursion limit might be an issue for large n, but we use iterative or increase limit
        # n=18 -> 153 edges. Recursion depth 153 is fine.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == num_edges:
                return True
            
            i, j = edges[idx]
            
            # Symmetry breaking for the first few edges
            # We can fix the colors of the first few edges to break color permutation symmetry.
            # Edge (0,1) can be 0.
            # Edge (0,2) can be 0 or 1.
            # Edge (0,3) can be 0, 1, or 2.
            # Edge (0,4) can be 0, 1, 2, or 3.
            # This reduces search space significantly.
            
            if i == 0 and j == 1:
                color_range = [0]
            elif i == 0 and j == 2:
                color_range = [0, 1]
            elif i == 0 and j == 3:
                color_range = [0, 1, 2]
            elif i == 0 and j == 4:
                color_range = [0, 1, 2, 3]
            else:
                color_range = [0, 1, 2, 3]
                
            # Shuffle colors for non-symmetry-breaking edges to find solutions faster?
            # Or keep deterministic? Deterministic is safer for reproducibility, 
            # but randomization can help escape local minima in backtracking order.
            # Let's try a deterministic order first, maybe with some heuristic.
            # Actually, for exact search, order doesn't matter for completeness, only for speed.
            # We'll stick to 0,1,2,3 for now.
            
            for c in color_range:
                # Check if adding edge (i,j) with color c creates a C6 in color c
                if has_path_len_5(adj[c], i, j):
                    continue
                
                # Assign
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[i][j] = c
                colors[j][i] = c
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[i][j] = -1
                colors[j][i] = -1
                
            return False

        if backtrack(0):
            # Flatten colors
            res = []
            for i in range(n):
                for j in range(i + 1, n):
                    res.append(colors[i][j])
            return res
        return None

    # Strategy:
    # 1. Try to find a solution for n=18.
    # 2. If fails, try n=17.
    # 3. If fails, return a known valid n=17 or n=16 coloring.
    
    # We allocate time dynamically.
    # n=18 is the target. If we find it, score is high.
    # n=17 is the baseline.
    
    # Time allocation:
    # Try n=18 with ~50s.
    # Try n=17 with ~40s (though n=17 is likely to be found very quickly by this solver).
    # Fallback.
    
    # Let's run n=18 first.
    # Note: The solver is deterministic. If it fails, it fails.
    # We might need multiple restarts with different random seeds for symmetry breaking?
    # The symmetry breaking above is fixed. 
    # To add diversity, we can randomize the order of colors tried for non-fixed edges.
    
    def solve_with_randomization(n, time_budget, seed):
        random.seed(seed)
        deadline = time.time() + time_budget
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        edge_idx = 0
        num_edges = len(edges)
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == num_edges:
                return True
            
            i, j = edges[idx]
            
            # Symmetry breaking
            if i == 0 and j == 1:
                color_range = [0]
            elif i == 0 and j == 2:
                color_range = [0, 1]
            elif i == 0 and j == 3:
                color_range = [0, 1, 2]
            elif i == 0 and j == 4:
                color_range = [0, 1, 2, 3]
            else:
                color_range = [0, 1, 2, 3]
                random.shuffle(color_range)
            
            for c in color_range:
                if has_path_len_5(adj[c], i, j):
                    continue
                
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[i][j] = c
                colors[j][i] = c
                
                if backtrack(idx + 1):
                    return True
                
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[i][j] = -1
                colors[j][i] = -1
                
            return False

        if backtrack(0):
            res = []
            for i in range(n):
                for j in range(i + 1, n):
                    res.append(colors[i][j])
            return res
        return None

    # Try n=18
    # We have ~115s. Let's give n=18 about 60s.
    if time.time() < hard_deadline - 50:
        # Try a few random restarts for n=18
        for seed in range(5):
            if time.time() > hard_deadline - 50:
                break
            res = solve_with_randomization(18, 10, seed)
            if res:
                return {'n': 18, 'colors': res}

    # Try n=17
    # n=17 should be found very quickly.
    if time.time() < hard_deadline - 20:
        # Deterministic solve for n=17 is likely fast
        res = solve(17, 50)
        if res:
            return {'n': 17, 'colors': res}
        
        # If deterministic failed (unlikely), try randomized
        for seed in range(10):
            if time.time() > hard_deadline - 10:
                break
            res = solve_with_randomization(17, 5, seed)
            if res:
                return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 5:
        res = solve(16, 10)
        if res:
            return {'n': 16, 'colors': res}

    # Fallback: Known valid coloring for n=9 or n=10
    # The seed program had a fallback for n=9.
    # Let's use a simple valid coloring for n=10 to be safe.
    # Actually, we can just return n=9 from the seed.
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    # Verify this quickly? The seed program assumed it was valid.
    # min(j-i, n-j+i) - 1
    # For n=9:
    # (0,1): min(1, 8)-1 = 0
    # (0,2): min(2, 7)-1 = 1
    # ...
    # This is a distance-based coloring.
    
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found n={res['n']}")
