import time
import sys

def construct():
    """
    Uses a constructive backtracking search to build a 4-coloring of K_n
    without monochromatic triangles.
    Builds the graph vertex by vertex, ensuring no monochromatic triangle
    is formed at each step.
    """
    start_time = time.time()
    time_limit = 100.0

    def solve(n):
        # c[i][j] stores color of edge (i, j) for i > j
        # We will store this in a list of lists for fast access
        # c[i] is a list of length i, c[i][j] is color of (i, j)
        c = [None] * n
        c[0] = []
        
        # Recursive backtracking
        # To avoid recursion depth issues and overhead, we can use a stack or just recursion if n is small.
        # n=50 is small enough for recursion.
        
        import sys
        sys.setrecursionlimit(10000)

        def backtrack(i):
            if i == n:
                return True
            
            # Assign colors to edges (i, j) for j in 0..i-1
            # We can process j in any order. Let's do j = i-1 down to 0 or 0 to i-1.
            # j = i-1 has the most constraints? No, j=0 has no constraints from k<j.
            # Actually, for a fixed i, the constraint for (i, j) depends on c[i][k] and c[j][k] for k < j.
            # So if we assign j in increasing order, we can use the already assigned c[i][k] for k < j.
            
            # Heuristic: Order j such that we pick the one with fewest valid colors first?
            # Or just iterate j = 0 to i-1.
            
            # To speed up, we can pre-compute the set of forbidden colors for each j.
            # But c[i][k] changes as we assign.
            
            # Let's try simple ordering: j = 0 to i-1.
            # Wait, if we assign j=0, no constraints.
            # j=1: constraint from k=0. c[i][0] and c[1][0].
            # j=2: constraints from k=0, 1.
            
            # We need to assign all j in 0..i-1.
            # This is a sub-problem.
            
            # Better: Use a helper to assign all edges for vertex i.
            # We can use backtracking within this step too.
            
            # Let's try a simpler approach:
            # For each j in 0..i-1, determine valid colors.
            # If we just iterate j=0..i-1, we might paint ourselves into a corner.
            # Example: c[i][0]=0. c[i][1]=0. c[1][0]=0. Triangle (i,1,0) is mono.
            # So we must check.
            
            # Let's implement a recursive function for the inner loop as well?
            # Or just try all 4^i combinations? No, 4^50 is too big.
            
            # We need to backtrack within the assignment of vertex i's edges.
            
            def assign_vertex(i, j):
                if j == i:
                    return True
                
                # Try colors 0..3 for edge (i, j)
                for col in range(4):
                    valid = True
                    # Check constraints: for all k < j, c[i][k] != col OR c[j][k] != col
                    # c[i][k] is already set for k < j (if we process j increasing)
                    # c[j][k] is already set (from previous vertices)
                    
                    for k in range(j):
                        if c[i][k] == col and c[j][k] == col:
                            valid = False
                            break
                    
                    if valid:
                        c[i][j] = col
                        if assign_vertex(i, j + 1):
                            return True
                        c[i][j] = None # Backtrack
                            
                return False

            if c[i] is None:
                c[i] = [None] * i
            
            return assign_vertex(i, 0)

        # Start backtracking from vertex 1 (vertex 0 has no edges)
        if backtrack(1):
            # Convert to flat list
            colors = []
            for i in range(n):
                for j in range(i):
                    colors.append(c[i][j])
            return colors
        return None

    best_n = 32
    best_colors = None

    # Try to find a solution for n=50, 48, ...
    for n in [50, 48, 46, 44, 42, 40, 38, 36, 34, 33]:
        if time.time() - start_time > time_limit:
            break
            
        # Set a tighter time limit per n to allow trying multiple n
        # But backtracking might be fast if it finds a solution quickly.
        # If it fails, it might take a long time to prove unsatisfiability.
        # We should probably add a node limit or time check inside backtrack.
        
        # For now, let's just try. If it's too slow, it will timeout.
        # To prevent hanging, we can add a global check.
        
        # Let's add a simple counter or time check in assign_vertex?
        # Modifying the nested function to check time is messy.
        # Let's rely on the fact that if n=50 is hard, it will fail fast or slow.
        # If it's slow, we might waste time.
        
        # Improvement: Add a randomization or different ordering?
        # Let's try n=32 first to ensure we have a valid solution, then try larger.
        
        pass

    # Actually, the above logic has a flaw: if backtrack(50) is too slow, we don't try 48.
    # We need a timeout mechanism inside the search.
    
    # Let's rewrite solve to include a time limit.
    
    def solve_with_timeout(n, limit):
        c = [None] * n
        c[0] = []
        start = time.time()
        
        def backtrack(i):
            if time.time() - start > limit:
                return False
            if i == n:
                return True
            
            if c[i] is None:
                c[i] = [None] * i
            
            def assign_vertex(j):
                if time.time() - start > limit:
                    return False
                if j == i:
                    return True
                
                for col in range(4):
                    valid = True
                    for k in range(j):
                        if c[i][k] == col and c[j][k] == col:
                            valid = False
                            break
                    
                    if valid:
                        c[i][j] = col
                        if assign_vertex(j + 1):
                            return True
                        c[i][j] = None
                            
                return False

            return assign_vertex(0)

        if backtrack(1):
            colors = []
            for i in range(n):
                for j in range(i):
                    colors.append(c[i][j])
            return colors
        return None

    # Try n=50 with 50s, then 48 with 20s, etc.
    # This is a bit arbitrary.
    
    # Let's try a simpler strategy:
    # 1. Try n=32 (should be very fast).
    # 2. Try n=40 (might be fast).
    # 3. Try n=50 (might be slow).
    
    # We want to maximize n.
    # We'll try 50, 48, ... 32.
    # If 50 fails (timeout), we try 48.
    
    remaining_time = time_limit - (time.time() - start_time)
    
    for n in [50, 48, 46, 44, 42, 40, 38, 36, 34, 33]:
        if remaining_time <= 5:
            break
            
        # Allocate time proportionally or fixed?
        # Let's give 10s to each attempt.
        t_alloc = min(10.0, remaining_time)
        
        colors = solve_with_timeout(n, t_alloc)
        if colors is not None:
            best_n = n
            best_colors = colors
            break
            
        remaining_time -= t_alloc

    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    else:
        # Fallback to n=32 algebraic construction from seed
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
