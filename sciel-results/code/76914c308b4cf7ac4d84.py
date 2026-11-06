import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback seed
    n_seed = 9
    # Construct a valid coloring for n=9
    # Simple: Cycle graph C9 is bipartite? No.
    # We just need a valid one.
    # Let's use the backtracking result if we find one, else generate a small one.
    
    best_n = 9
    best_colors = None
    
    # Generate seed colors for n=9
    # Edges in lexicographic order
    edges = [(i, j) for i in range(9) for j in range(i + 1, 9)]
    # A known valid coloring for n=9?
    # Actually, we can just run the local search for n=9 to get a valid one quickly.
    
    for n in range(17, 9, -1):
        if time.time() > deadline - 2:
            break
            
        # Allocate time
        if n >= 17:
            budget = 30.0
        elif n == 16:
            budget = 20.0
        elif n == 15:
            budget = 10.0
        elif n == 14:
            budget = 5.0
        else:
            budget = 2.0
            
        target_end = time.time() + budget
        res = _local_search(n, target_end)
        if res is not None:
            best_n = n
            best_colors = res
            # If we found a high n, we can return immediately if n is high enough?
            # The task says "Maximize n". So we should try 17, then 16, etc.
            # If we find 17, return it.
            if n == 17:
                return {'n': n, 'colors': res}
            # If we find 16, we might still try 17? No, we are looping down.
            # So if we find 17, we return. If we fail 17, we try 16.
            # If we find 16, we return it? Or do we try 17 again?
            # The loop structure is: try 17. If fail, try 16.
            # So if _local_search(17) returns None, we try 16.
            # If _local_search(16) returns colors, we should return it immediately?
            # Yes, because 16 > 15.
            return {'n': n, 'colors': res}
            
    # If we couldn't find n>=10, return a hardcoded small one or run search for 9
    if best_colors is None:
         res = _local_search(9, time.time() + 5)
         if res:
             return {'n': 9, 'colors': res}
         # Hardcoded fallback
         # n=3, K3 can be colored 0,0,0 (no C6)
         return {'n': 3, 'colors': [0, 0, 0]}

def _local_search(n, deadline):
    # Create edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Adjacency matrices for 4 colors
    # M[c][u][v] = 1 if (u,v) is color c
    M = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
    
    # Current colors array
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Initialize M
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        M[c][i][j] = 1
        M[c][j][i] = 1
        
    # Precompute A^k? No, compute on the fly.
    # Function to compute total C6 count
    def get_violations():
        total = 0
        for c in range(4):
            A = M[c].astype(np.int64)
            # Trace of A^6 / 12
            # A^2, A^3, A^6
            A2 = A @ A
            A3 = A2 @ A
            A6 = A3 @ A3
            total += np.trace(A6) // 12
        return total
        
    current_violations = get_violations()
    if current_violations == 0:
        return colors[:]
        
    best_colors = colors[:]
    best_violations = current_violations
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if current_violations == 0:
            return colors[:]
            
        # Pick a random edge to flip
        idx = random.randint(0, num_edges - 1)
        i, j = edges[idx]
        c_old = colors[idx]
        
        best_c_new = c_old
        best_v_new = current_violations + 1000 # Penalty for no move
        
        # Try other colors
        for c_new in range(4):
            if c_new == c_old:
                continue
                
            # Apply move
            M[c_old][i][j] = 0
            M[c_old][j][i] = 0
            M[c_new][i][j] = 1
            M[c_new][j][i] = 1
            
            v_new = get_violations()
            
            # Revert
            M[c_new][i][j] = 0
            M[c_new][j][i] = 0
            M[c_old][i][j] = 1
            M[c_old][j][i] = 1
            
            if v_new < best_v_new:
                best_v_new = v_new
                best_c_new = c_new
                
        # Accept best move
        if best_c_new != c_old:
            # Apply best move
            M[c_old][i][j] = 0
            M[c_old][j][i] = 0
            M[best_c_new][i][j] = 1
            M[best_c_new][j][i] = 1
            colors[idx] = best_c_new
            current_violations = best_v_new
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                
        temp *= cooling
        
    # Return best found
    if best_violations == 0:
        return best_colors
    else:
        # Check if best_colors is actually valid?
        # If best_violations > 0, it's invalid.
        # We should only return if valid.
        # Let's re-verify best_colors
        # If it's not valid, return None
        # But get_violations is expensive.
        # We can trust the count if it was 0.
        return None

# Optimization: The get_violations function is called 3 times per step (4 colors - 1).
# This might be slow in Python/NumPy overhead.
# But n=17 is small.
# Let's add a quick check: if we find 0 violations, return immediately.

# To be safe, if local search fails to find 0 violations, it returns None.
# The outer loop will try smaller n.

# One more thing: The seed solution n=9 must be valid.
# The local search for n=9 should easily find 0 violations.
