import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 118.0  # Keep 2s buffer for 120s limit

    # 1. Construct Polarity Graph of PG(2,5)
    q = 5
    n0 = q * q + q + 1

    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))

    n = len(points)

    def is_orthogonal(p, qpt):
        x1, y1, z1 = p
        x2, y2, z2 = qpt
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0

    # Build adjacency using bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if is_orthogonal(points[i], points[j]):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Build conflict bitmasks: conflict[i] has bit j if adj[i] & adj[j] != 0
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & adj[j]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    def has_independent_set_size_k(adj_bits, k, n_v):
        """Check if there's an independent set of size k using bitmasks."""
        if k > n_v:
            return False
        if k == 1:
            return n_v >= 1
        if k == 0:
            return True

        # Use recursive branch and bound with bitmasks
        # remaining: bitmask of available vertices
        # We need to find k vertices from remaining that are mutually non-adjacent

        def dfs(remaining, count):
            if count == k:
                return True
            # Upper bound: can we pick enough?
            if count + bin(remaining).count('1') < k:
                return False
            if remaining == 0:
                return False

            # Pick a vertex to branch on
            # Choose vertex with highest degree in remaining for better pruning
            # Or just pick lowest set bit for simplicity
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1

            # Option 1: Include v
            # Remove v and all its neighbors from remaining
            new_rem_include = remaining & ~(v_bit | adj_bits[v])
            if dfs(new_rem_include, count + 1):
                return True

            # Option 2: Exclude v
            new_rem_exclude = remaining & ~v_bit
            if dfs(new_rem_exclude, count):
                return True

            return False

        return dfs((1 << n_v) - 1, 0)

    def has_independent_set_size_k_fast(adj_bits, k, n_v):
        """Faster check using iterative approach with pruning."""
        if k > n_v:
            return False
        if k <= 1:
            return n_v >= k

        # Try to find independent set of size k
        # Use a greedy + backtracking hybrid
        # First, get a quick upper bound

        # Simple upper bound: n_v
        # Better: use the fact that in C4-free graphs, alpha is bounded

        # Let's use a more efficient recursive search
        # Order vertices by degree (descending) for better pruning

        # Build degree list
        degrees = [bin(adj_bits[i] & ((1 << n_v) - 1)).count('1') for i in range(n_v)]
        # Sort vertices by degree descending
        order = sorted(range(n_v), key=lambda x: -degrees[x])

        # Use the ordered list for branching
        # remaining_mask: which vertices are still available
        # picked: how many we've picked so far

        # To make this fast, we'll use a simple recursive approach
        # with the key optimization of trying high-degree vertices first

        # Actually, for the independence number, we want to find k non-adjacent vertices.
        # A vertex with high degree is "bad" to include (removes many options).
        # So for finding independent sets, we should try LOW degree vertices first?
        # No - if we include a low degree vertex, it removes few others.
        # If we exclude a high degree vertex, we keep more options.
        # Strategy: try including low-degree vertices first.

        # Let me reconsider. For maximum independent set:
        # - Including a vertex v removes N(v) from consideration
        # - We want to maximize the count
        # - Branch on a vertex: include or exclude
        # - For pruning: if remaining vertices < needed, fail

        # Let's just use the simple version but with a time limit
        # and a greedy lower bound check

        # Quick check: if even the greedy algorithm can't find k, 
        # it doesn't mean alpha < k, but it's a hint.

        # For now, use the straightforward DFS with bitmasks
        # which should be fast enough for n <= 43, k = 11

        def dfs(remaining, count):
            if count == k:
                return True
            if remaining == 0:
                return False
            # Pruning: not enough vertices left
            if count + bin(remaining).count('1') < k:
                return False

            # Pick a vertex - use lowest set bit
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1

            # Include v: remove v and its neighbors
            rem_without_v_and_nbs = remaining & ~(v_bit | adj_bits[v])
            if dfs(rem_without_v_and_nbs, count + 1):
                return True

            # Exclude v
            rem_without_v = remaining & ~v_bit
            if dfs(rem_without_v, count):
                return True

            return False

        return dfs((1 << n_v) - 1, 0)

    def try_add_vertex(adj_bits, conflict_bits, S_mask, n_cur):
        """Try to add a new vertex connected to vertices in S_mask.
        Returns (new_adj_bits, new_conflict_bits) if valid, else (None, None)."""
        # Check C4-freeness: S must be independent in conflict graph
        # i.e., no two vertices in S should have a common neighbor
        # This means for all u, v in S: conflict_bits[u] & (1<<v) should be 0
        # Equivalently, S should be an independent set in the conflict graph

        # Check: for each u in S, S should not intersect conflict_bits[u] (excluding u itself)
        # Actually, S is independent in conflict means: for all u,v in S, u!=v: v not in conflict[u]
        # Which means: for all u in S: conflict_bits[u] & S_mask == 0 (since u is not in conflict[u])

        # Wait, conflict[u] contains v if u and v share a common neighbor.
        # S is independent in conflict graph means no two vertices in S are adjacent in conflict graph.
        # So for all u in S: conflict_bits[u] & S_mask should have no bits from S except possibly u itself.
        # But u is not in conflict[u] (no self-loops), so we just check:
        # for all u in S: conflict_bits[u] & S_mask == 0

        temp = S_mask
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            if conflict_bits[v] & S_mask:
                return None, None
            temp ^= v_bit

        # Now check alpha: new graph has n_cur+1 vertices
        # New vertex v_new is connected to S
        # alpha(new_graph) = max(alpha(old_graph), 1 + alpha(old_graph \ S))
        # We need alpha(new_graph) <= 10
        # So we need: 1 + alpha(old_graph \ S) <= 10, i.e., alpha(old_graph \ S) <= 9
        # Equivalently, there is NO independent set of size 10 in old_graph \ S

        # Check if there's an independent set of size 10 in the subgraph induced by V\S
        # V\S has n_cur - popcount(S_mask) vertices
        # We need to check if alpha of that subgraph >= 10

        # Build the subgraph: vertices not in S
        # Map old indices to new indices
        # This is complex with bitmasks. Let's just check the full new graph.

        # Actually, let's check: does the new graph have an independent set of size 11?
        # New graph: n_cur + 1 vertices
        # New adjacency: old adj + new vertex connected to S
        # An independent set of size 11 in new graph either:
        # (a) doesn't include new vertex: independent set of size 11 in old graph (we know alpha(old) <= 10)
        # (b) includes new vertex: 1 + independent set of size 10 in old_graph \ S

        # So we just need to check: is there an independent set of size 10 in old_graph \ S?

        # Let's build the subgraph bitmask
        # Vertices in V\S
        S_complement = ((1 << n_cur) - 1) & ~S_mask
        n_sub = bin(S_complement).count('1')

        if n_sub < 10:
            # Can't have independent set of size 10
            pass
        else:
            # Need to check if alpha(subgraph) >= 10
            # Build subgraph adjacency
            # Map: old vertex i (in S_complement) -> new index
            sub_adj = []
            for i in range(n_cur):
                if S_complement & (1 << i):
                    # Get neighbors of i that are also in S_complement
                    sub_adj.append(adj_bits[i] & S_complement)
                else:
                    sub_adj.append(0)

            # Now check if subgraph has independent set of size 10
            if has_independent_set_size_k(sub_adj, 10, n_sub):
                return None, None

        # Valid! Build new adjacency and conflict
        new_n = n_cur + 1
        new_adj = [0] * new_n
        for i in range(n_cur):
            new_adj[i] = adj_bits[i]
            if S_mask & (1 << i):
                new_adj[i] |= (1 << new_n)
        new_adj[new_n] = S_mask

        # Build new conflict
        new_conflict = [0] * new_n
        for i in range(n_cur):
            new_conflict[i] = conflict_bits[i]
            # Check if i conflicts with new vertex
            # Common neighbors of i and new_vertex:
            # N(new_vertex) = S
            # N(i) = adj_bits[i] (plus new_vertex if i in S)
            # Common = S & (adj_bits[i] | (1<<new_n if i in S else 0))
            # = S & adj_bits[i] (since new_n not in S)
            # Wait, N(i) in new graph = adj_bits[i] U {new_n} if i in S
            # Common neighbors of i and new_n:
            # If i in S: N(i) = adj_bits[i] | (1<<new_n), N(new_n) = S
            #   Common = S & (adj_bits[i] | (1<<new_n)) = (S & adj_bits[i]) | (S & (1<<new_n))
            #   S & (1<<new_n) = 0, so common = S & adj_bits[i]
            # If i not in S: N(i) = adj_bits[i], N(new_n) = S
            #   Common = S & adj_bits[i]
            # So in both cases: common = S & adj_bits[i]
            if S_mask & adj_bits[i]:
                new_conflict[i] |= (1 << new_n)
                new_conflict[new_n] |= (1 << i)

        return new_adj, new_conflict

    # Main construction loop
    best_n = n
    best_adj = list(adj)
    best_conflict = list(conflict)

    def save_state():
        nonlocal best_n, best_adj, best_conflict
        best_n = n_cur
        best_adj = list(adj_bits)
        best_conflict = list(conflict_bits)

    n_cur = n
    adj_bits = list(adj)
    conflict_bits = list(conflict)
    save_state()

    while time.time() < deadline:
        added = False
        attempts = 0
        max_attempts = 200

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Generate a random candidate S
            # Strategy: pick a random subset of vertices
            # Try small sets first (size 1-3) for efficiency

            # Randomly choose size of S
            s_size = random.randint(1, min(3, n_cur))
            # Randomly choose s_size vertices
            candidates = list(range(n_cur))
            random.shuffle(candidates)
            S_list = candidates[:s_size]
            S_mask = 0
            for v in S_list:
                S_mask |= (1 << v)

            new_adj, new_conflict = try_add_vertex(adj_bits, conflict_bits, S_mask, n_cur)
            if new_adj is not None:
                adj_bits = new_adj
                conflict_bits = new_conflict
                n_cur += 1
                save_state()
                added = True
                break

        if not added:
            # Try with size 1 only
            for u in range(n_cur):
                if time.time() > deadline:
                    break
                S_mask = (1 << u)
                new_adj, new_conflict = try_add_vertex(adj_bits, conflict_bits, S_mask, n_cur)
                if new_adj is not None:
                    adj_bits = new_adj
                    conflict_bits = new_conflict
                    n_cur += 1
                    save_state()
                    added = True
                    break

        if not added:
            break

    # Build edge list from best state
    best_edges = []
    for u in range(best_n):
        mask = best_adj[u]
        while mask:
            v_bit = mask & (-mask)
            v = v_bit.bit_length() - 1
            if u < v:
                best_edges.append([u, v])
            mask ^= v_bit

    return {'n': best_n, 'edges': best_edges}
