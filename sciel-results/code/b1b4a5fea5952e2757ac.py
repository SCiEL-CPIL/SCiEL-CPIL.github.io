"""Rewritten program: K_{20,20}+1 init + Bitmask Simulated Annealing with incremental updates."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 40
    m = 401
    deadline = time.time() + 145  # safety margin before 150s kill

    # Use integer bitmasks for adjacency (fast intersection)
    adj_mask = [0] * n
    edge_set = set()

    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if u == v:
            return
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        edge_set.discard((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def book_size(u, v):
        """Number of common neighbors of u and v."""
        return bin(adj_mask[u] & adj_mask[v]).count('1')

    def compute_all_books():
        """Return dict edge -> book size."""
        books = {}
        for u, v in edge_set:
            books[(u, v)] = book_size(u, v)
        return books

    def get_metrics(books):
        if not books:
            return 0, 0
        max_bk = max(books.values())
        sum_sq = sum(b * b for b in books.values())
        return max_bk, sum_sq

    def k20_20_plus1_init():
        """Start with K_{20,20} (400 edges) + 1 intra-partition edge."""
        # Part A: vertices 0..19, Part B: vertices 20..39
        for i in range(20):
            for j in range(20):
                u = i
                v = 20 + j
                add_edge(u, v)
        # Add 1 edge within part A: (0, 1)
        add_edge(0, 1)
        # This gives 401 edges, with book of (0,1) = 20 (all of part B)

    # Initialize
    k20_20_plus1_init()
    books = compute_all_books()
    current_max, current_sum_sq = get_metrics(books)

    best_max = current_max
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)

    # Simulated Annealing Parameters
    T = 5.0
    cooling = 0.9995
    min_T = 0.05

    start_time = time.time()
    iterations = 0

    while time.time() < deadline:
        if T < min_T:
            break

        # Pick a random edge to remove
        u, v = random.choice(list(edge_set))

        # Pick a random non-edge to add
        added = False
        for attempt in range(20):
            c = random.randrange(n)
            d = random.randrange(n)
            if c == d:
                continue
            if c > d:
                c, d = d, c
            if (c, d) in edge_set:
                continue
            if (c, d) == (u, v):
                continue

            # Save state for potential revert
            old_adj_u = adj_mask[u]
            old_adj_v = adj_mask[v]
            old_adj_c = adj_mask[c]
            old_adj_d = adj_mask[d]

            # Tentative swap
            remove_edge(u, v)
            add_edge(c, d)

            # Incremental book update: only edges incident to {u,v,c,d} can change
            # Collect all edges that might have changed
            affected_vertices = set([u, v, c, d])
            changed_edges = set()
            for vert in affected_vertices:
                for neighbor in range(n):
                    if (vert, neighbor) in edge_set or (neighbor, vert) in edge_set:
                        a, b = (vert, neighbor) if vert < neighbor else (neighbor, vert)
                        changed_edges.add((a, b))

            # Recompute books only for changed edges
            new_books = dict(books)  # Copy
            for e in changed_edges:
                new_books[e] = book_size(e[0], e[1])

            new_max, new_sum_sq = get_metrics(new_books)

            # Acceptance criteria
            delta_max = new_max - current_max
            delta_sum = new_sum_sq - current_sum_sq

            accept = False
            if delta_max < 0:
                accept = True
            elif delta_max == 0:
                if delta_sum < 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta_sum / (50.0 * T))
                    accept = (random.random() < accept_prob)
            else:
                accept_prob = math.exp(-delta_max / T)
                accept = (random.random() < accept_prob)

            if accept:
                current_max = new_max
                current_sum_sq = new_sum_sq
                books = new_books
                if current_max < best_max or (current_max == best_max and current_sum_sq < best_sum_sq):
                    best_max = current_max
                    best_sum_sq = current_sum_sq
                    best_edge_set = set(edge_set)
                    best_adj_mask = list(adj_mask)
            else:
                # Revert
                remove_edge(c, d)
                add_edge(u, v)
                # Restore adj_mask (remove_edge/add_edge should handle it, but be safe)
                adj_mask[u] = old_adj_u
                adj_mask[v] = old_adj_v
                adj_mask[c] = old_adj_c
                adj_mask[d] = old_adj_d

            added = True
            break

        if not added:
            continue

        iterations += 1
        T *= cooling

    # Final polish: try to improve with very strict criteria
    polish_deadline = min(deadline, time.time() + 10)
    T_polish = 0.01
    while time.time() < polish_deadline:
        u, v = random.choice(list(best_edge_set))
        for attempt in range(20):
            c = random.randrange(n)
            d = random.randrange(n)
            if c == d:
                continue
            if c > d:
                c, d = d, c
            if (c, d) in best_edge_set:
                continue
            if (c, d) == (u, v):
                continue

            # Try this swap on a copy
            trial_edge_set = set(best_edge_set)
            trial_edge_set.discard((u, v))
            trial_edge_set.add((c, d))

            # Compute new max book
            trial_adj = [0] * n
            for a, b in trial_edge_set:
                trial_adj[a] |= (1 << b)
                trial_adj[b] |= (1 << a)

            new_max = 0
            for a, b in trial_edge_set:
                bk = bin(trial_adj[a] & trial_adj[b]).count('1')
                if bk > new_max:
                    new_max = bk

            if new_max < best_max:
                best_max = new_max
                best_edge_set = trial_edge_set
                break

    # Return the best solution found
    edge_list = list(best_edge_set)
    return [[u, v] for u, v in edge_list]
# EVOLVE-BLOCK-END
