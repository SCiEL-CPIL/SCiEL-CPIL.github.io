import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0

    def full_check(n, colors):
        """Verify no colour class has a C4."""
        # adj[i][c] = set of neighbors of i via color c
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i][c] & adj[j][c]) >= 2:
                        return False
        return True

    # Try the known Lazebnik-Woldar construction for n=26
    # This uses a finite geometry / algebraic construction
    # We'll try a direct construction based on GF(5) or similar
    # Actually, let's use a well-known construction:
    # For n=26, there's a construction based on the affine plane or similar
    
    # Let me try a systematic approach: backtracking with strong pruning
    
    def backtrack(n, time_limit):
        """
        Backtracking search for valid 5-coloring of K_n.
        Returns colors list or None.
        """
        num_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # edge_idx for quick lookup
        edge_idx = {}
        for idx, (i, j) in enumerate(edges):
            edge_idx[(i, j)] = idx
            edge_idx[(j, i)] = idx
        
        colors = [-1] * num_edges
        
        # For each color c and vertex v, track neighbors
        # neighbors[c][v] = set of vertices connected to v by color c
        neighbors = [[set() for _ in range(n)] for _ in range(5)]
        
        # For fast C4 checking: when adding edge (u,v) with color c,
        # check that no other vertex w has both u and v as neighbors in color c
        # i.e., |neighbors[c][u] & neighbors[c][v]| must be 0 after adding
        
        def try_assign(edge_pos, color):
            """Try to assign color to edge at position edge_pos.
            Returns True if valid, False if creates C4."""
            i, j = edges[edge_pos]
            # Check: in color `color`, do i and j have a common neighbor?
            common = neighbors[color][i] & neighbors[color][j]
            if len(common) >= 1:
                return False
            return True

        def assign(edge_pos, color):
            i, j = edges[edge_pos]
            colors[edge_pos] = color
            neighbors[color][i].add(j)
            neighbors[color][j].add(i)

        def unassign(edge_pos, color):
            i, j = edges[edge_pos]
            colors[edge_pos] = -1
            neighbors[color][i].discard(j)
            neighbors[color][j].discard(i)

        # Order edges for better pruning: use a fixed order but prioritize
        # edges that are more constrained
        # For now, use natural order but we can improve
        
        # Use a simple recursive backtracking with time check
        import sys
        sys.setrecursionlimit(10000)
        
        start_bt = time.time()
        found = [False]
        
        def dfs(pos):
            if time.time() - start_bt > time_limit:
                return False
            if found[0]:
                return True
            if pos == num_edges:
                found[0] = True
                return True
            
            i, j = edges[pos]
            # Try colors 0-4
            for c in range(5):
                if time.time() - start_bt > time_limit:
                    return False
                if try_assign(pos, c):
                    assign(pos, c)
                    if dfs(pos + 1):
                        return True
                    unassign(pos, c)
            return False

        # Try multiple random orderings / restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Shuffle edge order for diversity
            edge_order = list(range(num_edges))
            random.shuffle(edge_order)
            
            # Reset state
            for c in range(5):
                for v in range(n):
                    neighbors[c][v] = set()
            colors = [-1] * num_edges
            
            # Map: position in search -> actual edge index
            # We search in the shuffled order
            # For try_assign/assign/unassign, we need to work with actual edge indices
            # Let's restructure: search over positions 0..num_edges-1, 
            # where position p corresponds to edge_order[p]
            
            def try_assign2(search_pos, color):
                actual_pos = edge_order[search_pos]
                i, j = edges[actual_pos]
                common = neighbors[color][i] & neighbors[color][j]
                if len(common) >= 1:
                    return False
                return True

            def assign2(search_pos, color):
                actual_pos = edge_order[search_pos]
                i, j = edges[actual_pos]
                colors[actual_pos] = color
                neighbors[color][i].add(j)
                neighbors[color][j].add(i)

            def unassign2(search_pos, color):
                actual_pos = edge_order[search_pos]
                i, j = edges[actual_pos]
                colors[actual_pos] = -1
                neighbors[color][i].discard(j)
                neighbors[color][j].discard(i)

            def dfs2(search_pos):
                if time.time() - start_bt > time_limit:
                    return False
                if found[0]:
                    return True
                if search_pos == num_edges:
                    found[0] = True
                    return True
                
                for c in range(5):
                    if time.time() - start_bt > time_limit:
                        return False
                    if try_assign2(search_pos, c):
                        assign2(search_pos, c)
                        if dfs2(search_pos + 1):
                            return True
                        unassign2(search_pos, c)
                return False

            start_bt = time.time()
            found[0] = False
            if dfs2(0):
                return colors[:]
        
        return None

    # Try n values from high to low
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline:
            break
        
        # Allocate time for this n
        remaining = deadline - time.time()
        time_limit = min(remaining - 5.0, 15.0)
        if time_limit < 1.0:
            break
        
        result = backtrack(n, time_limit)
        if result is not None:
            # Verify
            if full_check(n, result):
                return {'n': n, 'colors': result}
    
    # Fallback: n=11 seed
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
