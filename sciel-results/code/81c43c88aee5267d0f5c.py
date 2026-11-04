import numpy as np
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    q = k
    # Calculate minimal d such that girth of CD(d, q) >= g
    # Girth of CD(d, q) is 2d + 2 for d >= 3.
    # We need 2d + 2 >= g => d >= g/2 - 1.
    # For g=10, d=4. For g=8, d=3.
    # Ensure d is at least 3 for the standard rule set, or handle smaller d.
    # The rule table provided in the seed handles c=2..d.
    # If g=6, d=2. Rule 2 is defined.
    # If g=4, d=1. Rule 1 not defined?
    # Task is g=10, so d=4.
    
    d = g // 2 - 1
    if d < 3:
        # Fallback to minimum supported d if g is very small
        # But for g=10, d=4, so this won't trigger.
        d = 3
        
    # Precompute rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # Convert rule to arrays for faster access
    rule_a = np.zeros(d+1, dtype=np.int64)
    rule_b = np.zeros(d+1, dtype=np.int64)
    for c in range(2, d+1):
        rule_a[c], rule_b[c] = rule[c]

    # GF(4) multiplication table
    if q == 4:
        gf4_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        gf4_table = None
    
    # BFS
    # Note: x_tuple has length d+1 (indices 0..d)
    # Index 0 is unused in the rule logic (rules start at c=2, using indices a,b which are < c).
    # Wait, rule[2] uses indices 1,1.
    # rule[3] uses indices 2,1.
    # rule[4] uses indices 1,2.
    # So indices 0 is never used in the recurrence?
    # Let's check the seed code:
    # start_vertex = (0, (0,) * (d + 1))
    # Y[:, 1] = np.arange(q)
    # Loop c from 2 to d.
    # Yes, index 0 is not touched in the neighbor generation loop.
    # It remains 0 for all generated vertices?
    # If index 0 is always 0, the effective dimension is d (indices 1..d).
    # Total vertices 2 * q^d.
    # For d=4, q=4: 2 * 4^4 = 512.
    # This is even better!
    
    start_vertex = (0, (0,) * (d + 1))
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    vertex_idx = 1
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        
        side, x_tuple = vertex
        x = np.array(x_tuple, dtype=np.int64)
        
        # Generate neighbors
        Y = np.zeros((q, d + 1), dtype=np.int64)
        Y[:, 1] = np.arange(q)
        
        for c in range(2, d + 1):
            a = rule_a[c]
            b = rule_b[c]
            if side == 0:
                if q == 4:
                    prod = gf4_table[Y[:, a], x[b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (Y[:, a] * x[b]) % q
                    Y[:, c] = (prod + x[c]) % q
            else:
                if q == 4:
                    prod = gf4_table[x[a], Y[:, b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (x[a] * Y[:, b]) % q
                    Y[:, c] = (x[c] - prod) % q
        
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            if side == 0:
                edges.append((visited[vertex], visited[neighbor]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
