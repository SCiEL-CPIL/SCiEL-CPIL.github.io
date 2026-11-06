"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin before 90s kill

    N = 256
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)

    # Numbers where 2a is a perfect square are forbidden (diagonal constraint)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers

    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:
                    adj[i] |= (1 << j)

    # Proper greedy coloring for upper bound on independent set size
    # Greedy coloring gives an upper bound because each color class is an independent set,
    # so the chromatic number >= alpha, and greedy gives an upper bound on chromatic number.
    # Wait, that's wrong. Greedy coloring gives an UPPER bound on chromatic number,
    # and chromatic number >= alpha (max independent set). So greedy colors >= alpha.
    # That means greedy_color_count is an upper bound on alpha. Good!
    
    def greedy_color_count(P):
        """Return number of colors in a greedy coloring of subgraph induced by P.
        This is an upper bound on the maximum independent set size in P."""
        if P == 0:
            return 0
        color_count = 0
        remaining = P
        while remaining:
            # Assign next color: find a maximal independent set in remaining
            color_class = 0
            temp = remaining
            while temp:
                # Pick vertex with lowest index (or could use degree heuristic)
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                # Check if v can be added to current color class (no neighbor in color_class)
                if (adj[v] & color_class) == 0:
                    color_class |= lb
                temp &= ~lb
            color_count += 1
            remaining &= ~color_class
        return color_count

    # Better vertex ordering: sort vertices by degree (highest first) for better pruning
    # We'll reorder vertices so that high-degree vertices come first
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda i: (-degrees[i], i))
    new_to_old = order  # new index -> old index
    old_to_new = [0] * M
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx

    # Rebuild adjacency in new order
    adj2 = [0] * M
    for i in range(M):
        old_i = new_to_old[i]
        mask = adj[old_i]
        # Convert to new indices
        new_mask = 0
        temp = mask
        while temp:
            lb = temp & (-temp)
            old_j = lb.bit_length() - 1
            new_j = old_to_new[old_j]
            new_mask |= (1 << new_j)
            temp &= ~lb
        adj2[i] = new_mask

    # Update idx_to_num for new ordering
    idx_to_num2 = [valid_numbers[new_to_old[i]] for i in range(M)]

    # Branch and bound with proper upper bound
    best_size = [0]
    best_mask = [0]
    nodes_explored = [0]

    def branch_and_bound(P, current_mask, current_size):
        if time.time() > deadline:
            return
        nodes_explored[0] += 1
        
        if P == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_mask[0] = current_mask
            return

        # Upper bound using greedy coloring
        ub = current_size + greedy_color_count(P)
        if ub <= best_size[0]:
            return

        # Branch on vertices in P
        # Use Tomita's strategy: branch on vertices in reverse order of coloring
        # For simplicity, just iterate through P
        temp = P
        while temp:
            if time.time() > deadline:
                return
            # Pick the highest-index vertex in P (last in coloring order)
            # Actually, let's pick the vertex with highest degree in P for better pruning
            # For now, just iterate from high to low index
            # Find highest set bit
            v = temp.bit_length() - 1
            lb = 1 << v
            
            # Add v to solution
            new_mask = current_mask | lb
            new_size = current_size + 1
            # New P: vertices in P that are not neighbors of v and not v
            new_P = P & ~adj2[v] & ~lb
            branch_and_bound(new_P, new_mask, new_size)
            
            temp &= ~lb

    # Initial P: all valid vertices
    all_mask = (1 << M) - 1
    branch_and_bound(all_mask, 0, 0)

    if best_size[0] > 0:
        result = [idx_to_num2[i] for i in range(M) if (best_mask[0] >> i) & 1]
    else:
        # Fallback: greedy
        result = []
        for x in valid_numbers:
            ok = True
            for y in result:
                if x + y in squares:
                    ok = False
                    break
            if ok:
                result.append(x)

    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]

    return sorted(result)
# EVOLVE-BLOCK-END
