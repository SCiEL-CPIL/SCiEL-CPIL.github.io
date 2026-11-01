"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # Select optimal q and d for LUW construction
    # For k=5, q=4 gives smaller graphs than q=5 for same girth
    if k == 5:
        q = 4
    else:
        q = k

    # Girth of LUW(q, d) is at least d + 5
    # We need girth >= g
    # So d + 5 >= g => d >= g - 5
    # Also d >= 2 for the construction to work well
    d = max(2, g - 5)
    
    # Ensure d is at least 3 for the rule definitions to be safe?
    # Seed used max(3, ...). Let's stick to d >= 2 if rules support it.
    # The rules in the seed are defined for c in range(2, d+1).
    # If d=2, c=2. rule[2]=(1,1). OK.
    # If d=3, c=2,3. OK.
    # If d=4, c=2,3,4. OK.
    
    # Precompute multiplication table for q=4
    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        mul_table = None

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
