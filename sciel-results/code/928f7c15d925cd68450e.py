"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    q = k
    # d is the parameter for the LUW construction
    # Girth is at least d + 5
    # We want girth >= g
    # So d + 5 >= g => d >= g - 5
    # The construction works for d >= 2
    # The seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9, g-5=4 (even), so d = 9-4=5?
    # Wait, the seed code:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # If g=9, (9-5)%2 == 0, so d = 9-4 = 5.
    # If d=5, girth is at least 5+5=10.
    # This is valid for girth 9.
    # The number of vertices will be larger.
    # Let's check the seed's d logic.
    # If g=9, d=5.
    # Size of CD(5,5): 2 * 5^((5+2)/2)? No, d is odd.
    # For d=5, size is 2 * 5^3 * 5^2? 
    # Actually, for d=5, the vectors have length 6.
    # The number of vertices is 2 * q^((d+1)/2) * q^((d+1)/2)?
    # Let's just use the seed's d calculation to be safe and consistent with the "design".
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    # Precompute multiplication table for efficiency, though for prime q, (a*b)%q is fast.
    # But vectorized operations on a table might be faster than element-wise mod in Python loops?
    # NumPy element-wise mod is fast.
    
    if q == 4:
        # GF(4) multiplication table
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        # For prime q, we can just use (a * b) % q
        # But let's stick to the table for uniformity if q is small, or just use mod.
        # Using mod is simpler and likely fast enough for q=5.
        pass

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)

    free_vals = np.arange(q, dtype=np.int64)

    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                # y[c] = x[c] + y[a] * x[b]
                # y[a] is the current column a in y_coords
                # x[b] is a scalar from x_arr
                prod = y_coords[:, a] * x_arr[b]
                if q != 4:
                    prod = prod % q
                else:
                    prod = mul_table[y_coords[:, a], x_arr[b]]
                y_coords[:, c] = (x_arr[c] + prod) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                # y[c] = x[c] - x[a] * y[b]
                prod = x_arr[a] * y_coords[:, b]
                if q != 4:
                    prod = prod % q
                else:
                    prod = mul_table[x_arr[a], y_coords[:, b]]
                y_coords[:, c] = (x_arr[c] - prod) % q
        return y_coords

    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((index[vertex], index[other_key]))

    return len(order), edges
# EVOLVE-BLOCK-END
