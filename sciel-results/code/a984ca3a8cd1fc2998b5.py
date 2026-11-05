import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    def solve_n(target_n):
        if time.time() > deadline:
            return None
            
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * target_n for _ in range(4)]
        
        # Edges in lexicographic order
        edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
        
        # Pre-allocate lists for speed? No, lists of ints are fine.
        
        def has_c6(u, v, c):
            # Check if adding edge (u,v) of color c creates a C6
            # This is true if there is a path of length 5 from u to v in adj[c]
            # Path: u - x1 - x2 - x3 - x4 - v
            
            # Level 1: neighbors of u
            level1 = adj[c][u]
            if not level1:
                return False
            
            # Level 2: neighbors of level1, excluding u and level1
            level2 = 0
            l1 = level1
            while l1:
                lsb = l1 & -l1
                idx = lsb.bit_length() - 1
                level2 |= adj[c][idx]
                l1 ^= lsb
            level2 &= ~(level1 | (1 << u))
            if not level2:
                return False
            
            # Level 3
            level3 = 0
            l2 = level2
            while l2:
                lsb = l2 & -l2
                idx = lsb.bit_length() - 1
                level3 |= adj[c][idx]
                l2 ^= lsb
            level3 &= ~(level2 | level1 | (1 << u))
            if not level3:
                return False
            
            # Level 4
            level4 = 0
            l3 = level3
            while l3:
                lsb = l3 & -l3
                idx = lsb.bit_length() - 1
                level4 |= adj[c][idx]
                l3 ^= lsb
            level4 &= ~(level3 | level2 | level1 | (1 << u))
            if not level4:
                return False
            
            # Level 5
            level5 = 0
            l4 = level4
            while l4:
                lsb = l4 & -l4
                idx = lsb.bit_length() - 1
                level5 |= adj[c][idx]
                l4 ^= lsb
            level5 &= ~(level4 | level3 | level2 | level1 | (1 << u))
            
            # Check if v is in level5
            return (level5 & (1 << v)) != 0

        # Backtracking
        # Symmetry breaking:
        # We can fix the first few edges to reduce search space.
        # c(0,1) = 0
        # c(0,2) = 0
        # c(0,3) = 0
        # c(0,4) = 0
        # c(0,5) = 0
        # This assumes a solution exists with these properties.
        # If not, we might miss it.
        # To be safe, we can try a few variations of symmetry breaking.
        # Or just use the standard loop and let it prune.
        
        # Let's try fixing c(0,1)=0, c(0,2)=0, c(0,3)=0.
        # This is usually safe for Ramsey graphs.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
                
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            
            # Determine valid colors
            # If we are in the symmetry breaking phase, restrict colors
            # For simplicity, let's just try all 4 colors.
            # To speed up, we can order colors by "least constraining" or just 0,1,2,3.
            
            for c in range(4):
                # Check if adding (u,v) with color c is valid
                if has_c6(u, v, c):
                    continue
                
                # Assign
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
            
            return False
        
        # Apply symmetry breaking manually before calling backtrack
        # Fix c(0,1) = 0
        # Fix c(0,2) = 0
        # Fix c(0,3) = 0
        # Fix c(0,4) = 0
        # Fix c(0,5) = 0
        
        # We can just set them and start backtrack from edge 5
        # But we must ensure they don't form a C6.
        # Since N is small, they won't.
        
        # Let's just run backtrack from 0, but with a twist:
        # If idx < 5 and u == 0, only allow c=0.
        # This enforces c(0,1)..c(0,5) = 0.
        
        def backtrack_sym(idx):
            if time.time() > deadline:
                return False
                
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            
            # Symmetry breaking
            if u == 0 and v <= 5:
                # Only try color 0
                if has_c6(u, v, 0):
                    return False
                adj[0][u] |= (1 << v)
                adj[0][v] |= (1 << u)
                if backtrack_sym(idx + 1):
                    return True
                adj[0][u] &= ~(1 << v)
                adj[0][v] &= ~(1 << u)
                return False
            
            for c in range(4):
                if has_c6(u, v, c):
                    continue
                
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack_sym(idx + 1):
                    return True
                
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
            
            return False

        if backtrack_sym(0):
            # Extract colors
            colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    for c in range(4):
                        if (adj[c][i] >> j) & 1:
                            colors.append(c)
                            break
            return {'n': target_n, 'colors': colors}
        return None

    best_result = None
    
    # Try N from 17 down to 10
    # 17 is the known best.
    for target_n in range(17, 9, -1):
        res = solve_n(target_n)
        if res:
            best_result = res
            break # Found the best, stop
        
        # If time is low, break
        if time.time() > deadline - 5:
            break

    if best_result is None:
        # Fallback: Try to find any valid coloring for small N
        # N=5 is trivial (all 0s)
        n = 5
        colors = [0] * (n * (n - 1) // 2)
        best_result = {'n': n, 'colors': colors}
        
    return best_result
