import time
import random

def construct():
    deadline = time.time() + 115
    n = 40
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def has_c6_with_edge(u, v, adj):
        """Check if adding (u,v) creates a C6 in graph with adjacency list adj."""
        a_mask = adj[u] & ~masks[v]
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit

            b_mask = adj[a] & ~masks[u] & ~masks[v]
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit

                c_mask = adj[b] & ~masks[u] & ~masks[a] & ~masks[v]
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit

                    d_mask = adj[c] & adj[v]
                    forbidden = masks[u] | masks[a] | masks[b] | masks[c]
                    if d_mask & ~forbidden:
                        return True
        return False

    def add_edge(u, v, adj):
        adj[u] |= masks[v]
        adj[v] |= masks[u]

    def remove_edge(u, v, adj):
        adj[u] &= ~masks[v]
        adj[v] &= ~masks[u]

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def greedy_fill(adj, edge_order=None):
        """Try to add all possible edges in the given order."""
        if edge_order is None:
            edge_order = all_edges[:]
            random.shuffle(edge_order)
        for u, v in edge_order:
            if (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v, adj):
                    add_edge(u, v, adj)

    def exhaustive_fill(adj):
        """Repeatedly try to add any addable edge until none remain."""
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if (adj[u] & masks[v]) == 0:
                    if not has_c6_with_edge(u, v, adj):
                        add_edge(u, v, adj)
                        changed = True

    def degree_order_fill(adj):
        """Fill edges in order of increasing degree sum of endpoints."""
        remaining = [e for e in all_edges if (adj[e[0]] & masks[e[1]]) == 0]
        while remaining:
            remaining.sort(key=lambda e: bin(adj[e[0]]).count('1') + bin(adj[e[1]]).count('1'))
            added_any = False
            new_remaining = []
            for u, v in remaining:
                if (adj[u] & masks[v]) == 0:
                    if not has_c6_with_edge(u, v, adj):
                        add_edge(u, v, adj)
                        added_any = True
                    else:
                        new_remaining.append((u, v))
                else:
                    new_remaining.append((u, v))
            remaining = new_remaining
            if not added_any:
                break

    def structured_init():
        """Create a structured initial graph with good density."""
        adj = [0] * n
        # Create 8 groups of 5 vertices each
        groups = [[i*5 + j for j in range(5)] for i in range(8)]
        # Add all edges within each group (K5)
        for group in groups:
            for i in range(5):
                for j in range(i+1, 5):
                    add_edge(group[i], group[j], adj)
        # Add some inter-group edges carefully
        for g1 in range(8):
            for g2 in range(g1+1, 8):
                # Add 1-2 edges between groups
                for _ in range(random.randint(1, 2)):
                    u = random.choice(groups[g1])
                    v = random.choice(groups[g2])
                    if not has_c6_with_edge(u, v, adj):
                        add_edge(u, v, adj)
        return adj

    best_adj = [0] * n
    best_edges = 0

    # Strategy 1: Multi-start with varied initializations
    while time.time() < deadline - 40:
        adj = [0] * n
        # Mix of random and structured initialization
        if random.random() < 0.3:
            adj = structured_init()
        else:
            # Random sparse initialization
            for _ in range(random.randint(15, 40)):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and (adj[u] & masks[v]) == 0:
                    if not has_c6_with_edge(u, v, adj):
                        add_edge(u, v, adj)
        
        # Use degree-ordered fill for better convergence
        if random.random() < 0.5:
            degree_order_fill(adj)
        else:
            greedy_fill(adj)
        
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj[:]

    # Strategy 2: Intensive local search with swap moves
    adj = best_adj[:]
    current_edges = best_edges
    no_improve_count = 0

    while time.time() < deadline - 15:
        improved = False

        # Phase 1: Try to add edges
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if (adj[x] & masks[y]) != 0:
                continue

            if not has_c6_with_edge(x, y, adj):
                add_edge(x, y, adj)
                current_edges += 1
                improved = True
                greedy_fill(adj)
                new_count = count_edges(adj)
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                current_edges = new_count
                no_improve_count = 0
                break

        # Phase 2: Try swap moves (remove one edge, add another)
        if not improved:
            # Find an edge to potentially remove
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] & masks[j])]
            if existing_edges:
                u_rem, v_rem = random.choice(existing_edges)
                remove_edge(u_rem, v_rem, adj)
                current_edges -= 1
                
                # Try to add a different edge
                added = False
                for _ in range(20):
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x == y or (adj[x] & masks[y]) != 0:
                        continue
                    if not has_c6_with_edge(x, y, adj):
                        add_edge(x, y, adj)
                        current_edges += 1
                        added = True
                        break
                
                greedy_fill(adj)
                new_count = count_edges(adj)
                
                if new_count >= best_edges:
                    if new_count > best_edges:
                        best_edges = new_count
                        best_adj = adj[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Revert if worse
                    # We can't easily revert, so just continue
                    no_improve_count += 1
                
                current_edges = new_count

        # Phase 3: Perturbation - remove multiple edges and re-fill
        if not improved and random.random() < 0.5:
            n_remove = random.randint(2, 4)
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] & masks[j])]
            random.shuffle(existing_edges)
            for u, v in existing_edges[:n_remove]:
                remove_edge(u, v, adj)
                current_edges -= 1
            
            greedy_fill(adj)
            new_count = count_edges(adj)
            if new_count > best_edges:
                best_edges = new_count
                best_adj = adj[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            current_edges = new_count

        if no_improve_count > 80:
            no_improve_count = 0
            # Reset to best and try again
            adj = best_adj[:]
            current_edges = best_edges

    # Final polish: exhaustive fill on best solution
    adj = best_adj[:]
    exhaustive_fill(adj)
    final_count = count_edges(adj)
    if final_count > best_edges:
        best_edges = final_count
        best_adj = adj[:]

    # One more polish pass with degree ordering
    adj = best_adj[:]
    degree_order_fill(adj)
    final_count2 = count_edges(adj)
    if final_count2 > best_edges:
        best_edges = final_count2
        best_adj = adj[:]

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] & masks[j]):
                edges.append([i, j])
    return edges
