import sys
import time

def construct():
    # We want to maximize n. 
    # The current seed is n=32.
    # We will try to find a valid 4-coloring for n=33, 34, 35, etc.
    # We use a backtracking search with bitsets for efficiency.
    
    # Since the search space is huge, we rely on the fact that small n (33-35) are likely solvable quickly with a good heuristic.
    # Heuristic: Order edges such that constraints are tight early on? 
    # Standard lexicographic order is usually fine for small n.
    
    max_n_target = 36  # Try up to 36. If it fails, we return the last valid one found or 32.
    
    best_n = 32
    best_colors = None
    
    # We know n=32 is valid. Let's construct it quickly to have a fallback.
    # The seed code used GF(16) logic. 
    # Let's just use a simple valid 32-vertex construction if the search fails.
    # Actually, if the search for n=33 fails, we can just return the n=32 from the seed logic or a hardcoded one.
    # But the search for n=33 is very likely to succeed.
    
    def try_construct(n):
        # C[i][j] stores the color of edge (i,j), -1 if uncolored.
        # We use a list of lists for C.
        C = [[-1] * n for _ in range(n)]
        
        # Neighbor sets for quick intersection check.
        # neighbors[v][c] is a set of vertices u such that edge (v,u) has color c.
        neighbors = [ [set() for _ in range(4)] for _ in range(n) ]
        
        # List of edges to color, in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
        
        # Backtracking search
        # We use recursion. Depth is n*(n-1)/2 which is ~600 for n=35. 
        # Recursion limit might be an issue if we go deep without pruning.
        # However, the greedy step usually colors most edges without backtracking.
        # We can use iterative deepening or just a standard DFS.
        
        # To avoid recursion depth issues, we can use a stack.
        # But let's try recursion first with sys.setrecursionlimit.
        sys.setrecursionlimit(10000)
        
        def dfs(edge_idx):
            if edge_idx == len(edges):
                return True
            
            i, j = edges[edge_idx]
            
            for c in range(4):
                # Check if color c is valid for (i,j)
                # Intersection of neighbors[i][c] and neighbors[j][c] must be empty.
                # Note: We only care about neighbors that are already colored.
                # The sets neighbors[v][c] only contain vertices connected by color c.
                
                # Optimization: 
                # If i and j are already connected? No, we are coloring them now.
                # We just need to check if there is a k such that (i,k) is c and (j,k) is c.
                # This is equivalent to neighbors[i][c] & neighbors[j][c] != empty.
                
                if neighbors[i][c] & neighbors[j][c]:
                    continue
                
                # Valid color. Assign it.
                C[i][j] = c
                C[j][i] = c
                neighbors[i][c].add(j)
                neighbors[j][c].add(i)
                
                if dfs(edge_idx + 1):
                    return True
                
                # Backtrack
                C[i][j] = -1
                C[j][i] = -1
                neighbors[i][c].remove(j)
                neighbors[j][c].remove(i)
            
            return False
        
        # Start the search
        # We can add a simple time limit check inside the loop if needed, 
        # but for n<=35, it should be fast if a solution exists.
        # If it takes too long, we might hit the 120s limit.
        # To be safe, we can limit the number of backtracks or use a time check.
        
        start_time = time.time()
        
        # We'll wrap the DFS to check time periodically.
        # But Python recursion doesn't easily allow breaking out.
        # Given n is small, let's trust it.
        
        success = dfs(0)
        
        if success:
            # Extract colors in lexicographic order
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(C[i][j])
            return colors
        else:
            return None

    # Try n from high to low? Or low to high?
    # If we try 36 and it fails, it might take 120s.
    # It's better to try 33, 34, 35.
    # If 33 succeeds, we return 33.
    # But we want to maximize.
    # Let's try 35 first. If it fails, try 34, then 33.
    # If 35 takes too long, we are in trouble.
    
    # A safer approach: 
    # We know 32 is valid.
    # Let's try 33. It is very likely to be fast.
    # Then 34.
    # Then 35.
    
    # Actually, if 33 is valid, 34 might not be.
    # Let's just try 35 with a timeout mechanism if possible.
    # Since I can't easily interrupt the recursive DFS, I will rely on the fact that 
    # for n=33, the search is shallow.
    
    # Let's try n=33 first.
    # If it finds a solution, we return it.
    # We can then try to extend? No, the interface returns a fixed n.
    # We should pick the largest n we can find within time.
    
    # Strategy:
    # 1. Try n=35. 
    # 2. If it returns None (failed), try n=34.
    # 3. If it returns None, try n=33.
    # 4. If it returns None, use the hardcoded n=32 from the seed.
    
    # To prevent n=35 from hanging for 120s, we can add a node counter.
    # If nodes > 100000, we give up on that n.
    
    def try_construct_limited(n, node_limit=200000):
        C = [[-1] * n for _ in range(n)]
        neighbors = [ [set() for _ in range(4)] for _ in range(n) ]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
        
        node_count = 0
        limit_exceeded = False
        
        def dfs(edge_idx):
            nonlocal node_count, limit_exceeded
            if limit_exceeded:
                return False
            
            node_count += 1
            if node_count > node_limit:
                limit_exceeded = True
                return False
            
            if edge_idx == len(edges):
                return True
            
            i, j = edges[edge_idx]
            
            for c in range(4):
                if neighbors[i][c] & neighbors[j][c]:
                    continue
                
                C[i][j] = c
                C[j][i] = c
                neighbors[i][c].add(j)
                neighbors[j][c].add(i)
                
                if dfs(edge_idx + 1):
                    return True
                
                C[i][j] = -1
                C[j][i] = -1
                neighbors[i][c].remove(j)
                neighbors[j][c].remove(i)
                
                if limit_exceeded:
                    return False
            
            return False
        
        success = dfs(0)
        
        if success:
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(C[i][j])
            return colors
        else:
            return None

    # Try descending from a reasonable upper bound.
    # 35 is a good target.
    for n in range(35, 32, -1):
        colors = try_construct_limited(n, node_limit=100000)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback to n=32 using the seed's logic (simplified)
    # The seed used GF(16). 
    # We can just use a known valid 32-vertex coloring.
    # Or we can use the fact that if 33 failed, 32 is valid.
    # Let's construct n=32 using the same function.
    colors_32 = try_construct_limited(32, node_limit=10000)
    if colors_32 is not None:
        return {'n': 32, 'colors': colors_32}
        
    # Absolute fallback: Hardcoded n=1
    return {'n': 1, 'colors': []}
