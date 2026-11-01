import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # The Lazebnik-Ustimenko-Woldar (LUW) graph construction.
    # For LUW(d, q), the graph is q-regular.
    # Girth is d + 5 if d is odd, d + 4 if d is even.
    # We need girth >= g.
    # For k=4, we set q=4.
    # We need to choose d such that girth(d, q) >= g.
    
    q = k
    
    # Determine d
    # If g <= 8, d=3 (girth 8) is enough.
    # If g == 9 or 10, d=5 (girth 10) is needed.
    # If g == 11 or 12, d=7 (girth 12) is needed.
    # Generally, if g is even, d = g - 5 (if g-5 is odd) or g - 4 (if g-5 is even)?
    # Let's just iterate d upwards from 3 to find the smallest d that satisfies the girth constraint.
    def luw_girth(d):
        return d + 5 if d % 2 == 1 else d + 4

    d = 3
    while luw_girth(d) < g:
        d += 1
    
    # Define field operations for GF(q)
    # Since q=4 (prime power 2^2), we can use specific tables or generic GF(4) logic.
    # For simplicity and speed, we use the standard representation for GF(4):
    # Elements: 0, 1, 2, 3.
    # 0 is zero. 1 is one.
    # Let alpha be a root of x^2 + x + 1 = 0. So alpha^2 = -alpha - 1.
    # In GF(4), 2 = alpha, 3 = alpha^2.
    # Addition is XOR: 0^0=0, 0^1=1, 0^2=2, 0^3=3, 1^1=0, 1^2=3, 1^3=2, 2^2=0, 2^3=1, 3^3=0.
    # Multiplication:
    # 0 * x = 0
    # 1 * x = x
    # 2 * 2 = 3
    # 2 * 3 = 1
    # 3 * 3 = 2
    # We can precompute tables for speed.
    
    add_table = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
    mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    
    def gf_add(a, b):
        return add_table[a][b]
        
    def gf_sub(a, b):
        return add_table[a][b] # In char 2, sub is add
        
    def gf_mul(a, b):
        return mul_table[a][b]

    # The LUW graph is defined on vertices (side, x0, x1, ..., xd)
    # side is 0 or 1.
    # x_i are in GF(q).
    # The edges connect (0, x) to (1, y) if:
    # y_1 = x_0
    # y_c = x_c + x_{c-1} * x_{c-d}  ?? No, the recurrence is specific.
    
    # According to Lazebnik et al. (1998), the LUW graph LUW(d, q) has vertices
    # v = (i, x_0, x_1, ..., x_d) where i in {0, 1} and x_j in GF(q).
    # Two vertices (i, x) and (1-i, y) are adjacent if:
    # y_0 = x_d
    # y_j = x_j + x_{j-1} x_{j-d} for j=1..d ? 
    # Actually, the condition is:
    # For side 0 to side 1:
    # y[0] = x[d]
    # y[j] = x[j] + x[j-1] * x[j-d]  for 1 <= j <= d-1 ?
    # Wait, the standard LUW definition:
    # Vertices are pairs (i, x) where x is a vector of length d+1?
    # Let's use the definition from the seed code which was provided and presumably correct for the family.
    
    # Seed code rule:
    # rule[c] = (a, b)
    # y[c] = x[c] + y[a] * x[b] (if side 0->1)
    # The seed code computes `rule` based on c.
    # Let's replicate the seed code's logic exactly to ensure correctness within the family.
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The seed used: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # This looks like a specific pattern for LUW.
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def get_neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is unused in seed?
        # Seed: `start = (0, (None,) + (0,) * d)`
        # So x has length d+1.
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = first
            # y[1] is implicitly 0?
            # In seed: y = [None, first] + [0] * (d - 1)
            # So y[0] is None, y[1] is first, y[2..d] are 0.
            # Then loop c from 2 to d.
            
            # We need to compute y[c] based on x and previous y's.
            # The recurrence in seed:
            # if side == 0: y[c] = x[c] + y[a] * x[b]
            # if side == 1: y[c] = x[c] - x[a] * y[b] (which is + in char 2)
            
            # We iterate c from 2 to d.
            # Note: y[1] is already set to 'first'.
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    val = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    val = gf_add(x[c], gf_mul(x[a], y[b]))
                y[c] = val
            yield (1 - side, tuple(y))

    # Build the graph via BFS from the start vertex
    start = (0, (0,) * (d + 1)) # Using 0 instead of None for index safety
    # Note: Seed used None for x[0]. We can use 0 as it's never accessed if we are careful,
    # but the rule indices a,b might access index 0?
    # rule[2] = (1,1). a=1, b=1.
    # rule[3] = (2,1).
    # rule[4] = (1,2).
    # rule[5]: c=5, r=0, c0=5. rule[5] = (1, 3).
    # rule[6]: c=6, r=1, c0=6. rule[6] = (5, 1).
    # Indices a and b range up to d.
    # Does any rule access index 0?
    # c0-2 for c=5 is 3.
    # c0-1 for c=5 is 4.
    # c0 for c=5 is 5.
    # 1 for c=5 is 1.
    # It seems indices are >= 1.
    # So x[0] is not accessed. We can safely use 0.
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # To speed up, we can precompute neighbors or just do BFS.
    # N is roughly 2 * q^d. For d=5, q=4, N=2048.
    # BFS is fast.
    
    for i in range(len(order)):
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
            
        if time.time() > deadline:
            break

    return len(order), edges
# EVOLVE-BLOCK-END
