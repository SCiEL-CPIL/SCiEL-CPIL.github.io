"""Cochromatic partition of a pinned random graph (n=14)."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 55  # safety margin before 60s limit
    edges = P['edges']
    edge_set = set()
    for a, b in edges:
        edge_set.add((min(a, b), max(a, b)))

    # Adjacency bitmask for fast clique/independent checks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    full_mask = (1 << n) - 1

    def is_clique(mask):
        # Check if vertices in mask form a clique
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # v must be adjacent to all other vertices in mask
            if (adj[v] & (mask ^ lsb)) != (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def is_independent(mask):
        # Check if vertices in mask form an independent set
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def is_valid_part(mask):
        return is_clique(mask) or is_independent(mask)

    # Precompute: for each subset, whether it's a valid part
    # There are 2^14 = 16384 subsets, precompute all
    valid_part = [False] * (1 << n)
    valid_part[0] = True
    for mask in range(1, 1 << n):
        if is_clique(mask) or is_independent(mask):
            valid_part[mask] = True

    # Lower bound: at least 2 (unless graph is complete or empty)
    # Upper bound: greedy gives some bound
    # Greedy: assign each vertex to existing part if valid, else new part
    def greedy_partition():
        parts = []
        for v in range(n):
            vbit = 1 << v
            placed = False
            for i, p in enumerate(parts):
                if valid_part[p | vbit]:
                    parts[i] = p | vbit
                    placed = True
                    break
            if not placed:
                parts.append(vbit)
        return len(parts), parts

    ub, _ = greedy_partition()

    # Try k from 2 to ub
    # Use DFS: assign each vertex to one of k parts, each part must be valid
    # We use a smarter approach: try to cover all vertices with k valid parts

    # For a given k, check if we can partition into k valid parts
    # Use recursive backtracking with pruning

    best_result = None

    def solve_k(k):
        """Try to find partition into exactly k valid parts."""
        # State: list of k masks (current parts), remaining vertices
        # We assign vertices one by one to minimize search
        # Order vertices by degree (high degree first for clique, low for indep)
        # Actually, let's use a different approach:
        # Recursively pick a valid part from remaining, remove it, recurse with k-1

        # Memoization: frozenset of remaining mask -> min parts needed
        # But that's expensive. Let's use simple backtracking with ordering.

        # Better approach: enumerate valid subsets of the remaining vertices
        # and try to cover with k parts

        # For k=2: need two valid parts whose union is full_mask and intersection is empty
        # i.e., mask and full_mask^mask are both valid parts

        if k == 1:
            if valid_part[full_mask]:
                return [full_mask]
            return None

        if k == 2:
            # Try all masks, check mask and complement are valid
            for mask in range(1, 1 << n):
                comp = full_mask ^ mask
                if mask == 0 or comp == 0:
                    continue
                if valid_part[mask] and valid_part[comp]:
                    return [mask, comp]
            return None

        # For k >= 3, use backtracking
        # Pick the first remaining vertex, try all valid parts containing it
        def backtrack(remaining, k_left, current_parts):
            if time.time() > deadline:
                return None
            if k_left == 0:
                if remaining == 0:
                    return current_parts
                return None
            if remaining == 0:
                return current_parts  # fewer parts than k_left, that's fine
            # Lower bound: need at least 1 part if remaining != 0
            if k_left < 1:
                return None

            # Pick a vertex from remaining (first one)
            lsb = remaining & (-remaining)
            v = lsb.bit_length() - 1

            # Find all valid parts that contain v and are subset of remaining
            # Enumerate subsets of remaining that contain v
            rest = remaining ^ lsb
            sub = rest
            while True:
                candidate = sub | lsb
                if valid_part[candidate]:
                    result = backtrack(remaining ^ candidate, k_left - 1, current_parts + [candidate])
                    if result is not None:
                        return result
                if sub == 0:
                    break
                sub = (sub - 1) & rest
            return None

        # Start backtracking
        result = backtrack(full_mask, k, [])
        return result

    # Try increasing k from lower bound
    # Lower bound: 2 (since graph is neither complete nor empty)
    for k in range(2, ub + 1):
        if time.time() > deadline:
            break
        result = solve_k(k)
        if result is not None:
            # Convert masks to lists
            parts = []
            for mask in result:
                verts = []
                for v in range(n):
                    if mask & (1 << v):
                        verts.append(v)
                parts.append(tuple(verts))
            return parts

    # Fallback: return greedy result
    _, greedy_parts = greedy_partition()
    parts = []
    for mask in greedy_parts:
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        parts.append(tuple(verts))
    return parts
# EVOLVE-BLOCK-END
