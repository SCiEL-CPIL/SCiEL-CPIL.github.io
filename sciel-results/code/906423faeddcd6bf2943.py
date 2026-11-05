import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Petersen Graph Definition
# Vertices 0..9
# Edges:
# Outer: (0,1), (1,2), (2,3), (3,4), (4,0)
# Inner: (5,6), (6,7), (7,8), (8,9), (9,5) -- Note: Standard Petersen inner star is usually (5,7), (7,9), (9,6), (6,8), (8,5) or similar. 
# Let's use the standard definition:
# Outer cycle: 0-1-2-3-4-0
# Inner star: 5-7-9-6-8-5
# Spokes: 0-5, 1-6, 2-7, 3-8, 4-9

_PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),          # Outer
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),          # Inner
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)           # Spokes
]

def _build_petersen_adj():
    adj = [[] for _ in range(10)]
    for u, v in _PETERSEN_EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj

_PETERSEN_ADJ = _build_petersen_adj()

# Pre-compute all non-backtracking closed walks of length L in [5, 15]
# Return a list of coefficient vectors (tuples of 15 ints)
def _generate_closed_walk_vectors(max_len=15):
    vectors = set()
    n_edges = 15
    
    # Map (u, v) to edge index and direction
    # We need a consistent orientation for each edge to define direction.
    # Let's say edge (u, v) with u < v is index i.
    # Traversal u->v is +1, v->u is -1.
    
    edge_map = {}
    for i, (u, v) in enumerate(_PETERSEN_EDGES):
        if u > v:
            u, v = v, u
        edge_map[(u, v)] = (i, 1)
        edge_map[(v, u)] = (i, -1)
        
    # DFS from each vertex to find closed walks
    # State: (current, prev, length, path_edges, path_dirs)
    # To avoid duplicates, we only record walks that start at the smallest vertex in the walk?
    # Or just record all and let the set handle duplicates?
    # The number of walks is small enough that we can just record all and dedupe by vector.
    
    for start in range(10):
        # stack: (current, prev, length, coeff_vec)
        # coeff_vec is a list of 15 ints, initialized to 0
        stack = [(start, -1, 0, [0]*n_edges)]
        
        while stack:
            curr, prev, length, coeff = stack.pop()
            
            # If length >= 5 and curr == start, we have a closed walk
            # Note: We must ensure the walk is non-backtracking.
            # The DFS logic ensures we don't go back to prev.
            # However, we might return to start via a path that visited start earlier?
            # Non-backtracking closed walk: no immediate reversal.
            # If we return to start, the previous vertex must not be the one we came from start initially?
            # Actually, standard definition: a closed walk is a sequence v0...vk=v0.
            # Non-backtracking means v_{i-1} != v_{i+1} for all i.
            # This implies v0 != v2, etc.
            # If we return to start, the step into start comes from some u. The step out of start initially was to w.
            # We must ensure u != w? No, that's not the condition.
            # The condition is local: at each step, you don't go back to where you came from.
            # So if we are at start, and we came from u, we cannot go to u.
            # But we are ending the walk.
            # The only constraint is that the walk itself is non-backtracking.
            # My DFS ensures `next_v != prev`.
            # So if I start at 0, go to 1, ... and come back to 0, the last step was from some u to 0.
            # The step before that was from w to u.
            # The first step was 0 to 1.
            # Is it possible that u == 1?
            # If the path was 0-1-...-1-0, then at the last step, we are at 1, prev is ...?
            # If the path is 0-1-0, length 2. Not allowed (L>=5).
            # If the path is 0-1-2-1-0, length 4. Not allowed.
            # For L>=5, can we have 0-1-...-1-0?
            # Yes, e.g., 0-1-2-3-1-0.
            # Step 1: 0->1. Prev -1.
            # Step 2: 1->2. Prev 0.
            # Step 3: 2->3. Prev 1.
            # Step 4: 3->1. Prev 2. (1 != 2, OK)
            # Step 5: 1->0. Prev 3. (0 != 3, OK)
            # This is a valid non-backtracking closed walk.
            # So we just need to record the coefficient vector.
            
            if length >= 5 and curr == start:
                vectors.add(tuple(coeff))
                
            if length == max_len:
                continue
                
            for next_v in _PETERSEN_ADJ[curr]:
                if next_v == prev:
                    continue
                # Update coefficient
                new_coeff = coeff[:]
                e_idx, direction = edge_map[(curr, next_v)]
                new_coeff[e_idx] += direction
                stack.append((next_v, curr, length + 1, new_coeff))
                
    return list(vectors)

# Pre-compute the constraint vectors
# We need walks of length 5 to g-1.
# For g=16, lengths 5..15.
_CONSTRAINT_VECTORS = _generate_closed_walk_vectors(15)
_C_MATRIX = np.array(_CONSTRAINT_VECTORS) if _CONSTRAINT_VECTORS else np.empty((0, 15))

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    if k != 3:
        # Fallback for other k (though task is k=3)
        return _trivial_regular(k, g)
        
    # Moore bound for k=3, g=16 is 512.
    # n = 10 * m.
    # 10 * m >= 512 => m >= 52.
    # Since we use Z_m, m should be an integer.
    # Primes are good for random search uniformity, but any m works.
    # Let's iterate m from 52 upwards.
    
    # If g is small, we might need smaller m.
    # Girth of Petersen is 5.
    # If g <= 5, m=1 works? No, m=1 gives 10 vertices, girth 5.
    # If g=6, m=1 fails (girth 5). m=2?
    
    min_m = 52 if g == 16 else max(1, (512 // 10) if g >= 16 else 1)
    # Actually, let's just start from 1 and go up, but stop if time runs out.
    # For g=16, we know m must be >= 52.
    # For smaller g, m can be smaller.
    
    # Let's determine min_m based on Moore bound for general k,g?
    # Moore bound M(k,g) for even g: k * (k-1)^(g/2 - 1)
    # For k=3, g=16: 3 * 2^7 = 384? 
    # Wait, Moore bound for (3,16) is often cited as 512 or 384?
    # The prompt says "best lower bound is 512".
    # So n >= 512.
    # n = 10 * m.
    # m >= 51.2 => m >= 52.
    
    if g == 16:
        min_m = 52
    else:
        # Calculate Moore bound roughly
        if g % 2 == 0:
            mb = k * (k-1)**(g//2 - 1)
        else:
            mb = k * (k-1)**((g-1)//2)
        min_m = max(1, int(np.ceil(mb / 10.0)))
        
    # If _C_MATRIX is empty (shouldn't happen for g>5), return trivial
    if len(_C_MATRIX) == 0:
        return (10, _PETERSEN_EDGES)
        
    # For each m, try random vectors
    # Number of constraints is len(_C_MATRIX).
    # For each m, probability of a random vector failing is approx N_constraints * (1/m)
    # If N_constraints is ~100 and m=52, prob fail ~ 100/52 > 1. So many fails.
    # We need to try many random vectors.
    
    best_result = None
    
    for m in range(min_m, 200): # Try up to m=200 (n=2000)
        if time.time() > DEADLINE:
            break
            
        # Try random assignments
        # Use numpy for speed
        n_tries = 50
        found = False
        for _ in range(n_tries):
            v = np.random.randint(0, m, size=15)
            
            # Check constraints: C @ v % m != 0
            # C shape (N, 15), v shape (15,)
            # Result shape (N,)
            sums = (_C_MATRIX @ v) % m
            
            if not np.any(sums == 0):
                # Valid
                # Construct graph
                n = 10 * m
                edges = []
                for idx, (u, val) in enumerate(_PETERSEN_EDGES):
                    # Wait, _PETERSEN_EDGES is list of (u,v)
                    pass # Placeholder
                
                # Re-build edges properly
                edges = []
                for i, (u, v_base) in enumerate(_PETERSEN_EDGES):
                    a = v[i]
                    for k_group in range(m):
                        su = u * m + k_group
                        sv = v_base * m + ((k_group + a) % m)
                        edges.append((su, sv))
                        
                best_result = (n, edges)
                found = True
                break
                
        if found:
            break
            
    if best_result is not None:
        return best_result
        
    # Fallback: Return a large valid graph or seed logic
    # If we didn't find one, return the Petersen graph (invalid for g=16) or a larger one?
    # The evaluator scores invalid results as 0.
    # We should try to return *something* valid if possible, even if large.
    # Or just return the seed's LUW construction if we have it?
    # The current program has _luw_fallback. I should include a robust fallback.
    
    return _luw_fallback(k, g)

def _trivial_regular(k, g):
    # Simple fallback
    if k == 3:
        return (10, _PETERSEN_EDGES)
    # For other k, create a cycle? Not k-regular.
    # Just return Petersen for k=3, else error?
    # Task is k=3.
    return (10, _PETERSEN_EDGES)

def _luw_fallback(k, g):
    # Reproduce the seed's Lazebnik-Ustimenko-Woldar construction as a safe fallback.
    # This code is from the current program.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
