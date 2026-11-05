import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        """Check if no triple of vertices has 3 or more common neighbors."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (cij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a violating triple and its common neighbors, or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k, cijk)
        return None

    def make_valid(adj):
        """Remove edges until the graph is valid."""
        attempts = 0
        while True:
            viol = find_violation(adj)
            if viol is None:
                return True
            i, j, k, cijk = viol
            # Remove an edge from one of the triple vertices to one of the common neighbors
            # Pick a random common neighbor and remove edge (i, x)
            x = random.choice([b for b in range(n) if cijk & (1 << b)])
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            attempts += 1
            if attempts > 5000:
                return False

    def greedy_add(adj):
        """Try to add edges greedily. Returns number of edges added."""
        added = 0
        improved = True
        while improved:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    added += 1
                    improved = True
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
        return added

    def local_search(adj, time_limit):
        """Local search: try swaps (remove one edge, add another) to unlock more additions."""
        edges_in = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
        
        best_count = count_edges(adj)
        
        while time.time() < time_limit:
            if not non_edges:
                break
            
            # Try random swaps
            random.shuffle(edges_in)
            random.shuffle(non_edges)
            
            found = False
            for e_rm in edges_in[:20]:
                for e_add in non_edges[:20]:
                    u1, v1 = e_rm
                    u2, v2 = e_add
                    # Check they're different
                    if (u1 == u2 and v1 == v2):
                        continue
                    # Also check we're not adding an edge that's already there
                    if adj[u2] & (1 << v2):
                        continue
                    
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    # Remove
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    # Add
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    if is_valid(adj):
                        # Try to add more edges after the swap
                        greedy_add(adj)
                        c = count_edges(adj)
                        if c > best_count:
                            best_count = c
                            # Update edges_in and non_edges
                            edges_in = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
                            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
                        found = True
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                if found:
                    break
            
            if not found:
                # No beneficial swap found, try a few more random restarts of the search
                break

        return best_count

    def generate_paley_19():
        adj = [0] * 19
        residues = set((x * x) % 19 for x in range(1, 10))
        for u in range(19):
            for v in range(u + 1, 19):
                if (v - u) % 19 in residues or (u - v) % 19 in residues:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        adj20 = adj[:] + [0]
        return adj20

    def generate_random(p):
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj

    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    random.seed(12345)

    # Multiple restarts with different strategies
    num_starts = 8
    for s in range(num_starts):
        if time.time() > deadline - 15:
            break

        # Choose initial graph
        if s == 0:
            adj = generate_paley_19()
        elif s == 1:
            adj = generate_random(0.6)
        elif s == 2:
            adj = generate_random(0.5)
        elif s == 3:
            adj = generate_random(0.7)
        else:
            adj = generate_random(random.uniform(0.3, 0.7))

        # Make valid
        if not make_valid(adj):
            continue

        # Greedy add
        greedy_add(adj)

        # Local search with time limit
        remaining = deadline - 15 - time.time()
        if remaining > 5:
            local_search(adj, time.time() + min(remaining * 0.7, 10))

        update_best(adj)

    # Final fallback
    if not best_edges:
        # Return a simple valid graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 2 == 1:  # bipartite-like
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # This might not be valid, so just return a path
        return [[i, i + 1] for i in range(n - 1)]

    return best_edges
