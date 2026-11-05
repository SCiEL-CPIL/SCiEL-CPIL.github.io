import random
import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill

    best_edges = []
    best_count = 0

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_pairs_set = set(all_pairs)

    def verify_edges(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            if u == v or u >= n or v >= n:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in [(a, b) if a < b else (b, a) for a, b in edge_list if (a, b) != (u, v) and (b, a) != (u, v)]:
                pass
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Check for K2,3: no pair of vertices can have > 2 common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True

    def can_add(u, v, adj_mask, common):
        # 1. Check if u,v already connected
        if (adj_mask[u] >> v) & 1:
            return False
        # 2. Check common neighbors of u and v
        if common[u][v] > 2:
            return False
        # 3. Check neighbors of v (excluding u)
        # For each w in N(v) \ {u}, new common[u][w] = old common[u][w] + 1 must be <= 2
        # So old common[u][w] <= 1
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if common[u][w] >= 2:
                return False
            m -= lsb
        # 4. Check neighbors of u (excluding v)
        # For each w in N(u) \ {v}, new common[v][w] = old common[v][w] + 1 must be <= 2
        # So old common[v][w] <= 1
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if common[v][w] >= 2:
                return False
            m -= lsb
        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append((u, v))
        # Update common counts
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove((u, v))
        # Update common counts (after removal)
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges, time_limit=None):
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if time_limit and time.time() > time_limit:
                break
            if can_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)

    def local_search(initial_edges, time_limit=None):
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in initial_edges:
            add_edge(u, v, adj_mask, common, edges)

        best_local_edges = list(edges)
        best_local_count = len(edges)

        # Try to add edges until no more can be added
        changed = True
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if time_limit and time.time() > time_limit:
                    break
                if can_add(u, v, adj_mask, common):
                    add_edge(u, v, adj_mask, common, edges)
                    changed = True
                    if len(edges) > best_local_count:
                        best_local_count = len(edges)
                        best_local_edges = list(edges)

        return best_local_edges, adj_mask, common, edges

    def build_initial_graph():
        """Build a dense initial graph using a structured approach."""
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []

        # Start with a 5-regular graph using a known construction
        # Use a circulant-like structure
        # For n=20, we can use offsets that avoid K2,3

        # Try to build a graph with high degree
        # Use a greedy approach with a good ordering

        # First, try to add all edges in a structured way
        # Use the fact that for K2,3-free, we need common[u][v] <= 2

        # Initialize with a 4-regular expander-like graph
        # Then greedily add more edges

        # Random regular graph initialization
        for _ in range(100):
            adj_mask_try = [0] * n
            common_try = [[0] * n for _ in range(n)]
            edges_try = []

            # Try to build a 5-regular graph
            targets = [5] * n
            for _ in range(50):
                # Find vertices with highest remaining degree
                unmet = [(targets[i], i) for i in range(n)]
                random.shuffle(unmet)
                unmet.sort(reverse=True)

                added = False
                for deg1, u in unmet:
                    if deg1 <= 0:
                        continue
                    # Find a vertex v to connect to
                    candidates = [v for v in range(n) if v != u and not ((adj_mask_try[u] >> v) & 1)]
                    random.shuffle(candidates)
                    for v in candidates:
                        if can_add(u, v, adj_mask_try, common_try):
                            add_edge(u, v, adj_mask_try, common_try, edges_try)
                            targets[u] -= 1
                            targets[v] -= 1
                            added = True
                            break
                    if added:
                        break
                if not added:
                    break

            if len(edges_try) > len(edges):
                edges = edges_try
                adj_mask = adj_mask_try
                common = common_try

        # Now greedily add more edges
        greedy_fill(adj_mask, common, edges)

        return adj_mask, common, edges

    # Main Search
    # Build initial graph
    current_adj, current_common, current_edges = build_initial_graph()
    current_edges, current_adj, current_common, _ = local_search(current_edges, time_limit=start_time + max_time * 0.3)

    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = [[u, v] for u, v in current_edges]

    # Iterated Local Search
    T = 1.0
    cooling = 0.9995

    while time.time() - start_time < max_time * 0.9:
        # Perturb
        kick_size = random.choice([1, 2, 2, 3, 3, 4, 5])
        if len(current_edges) > kick_size:
            to_remove = random.sample(current_edges, min(kick_size, len(current_edges)))
            for u, v in to_remove:
                remove_edge(u, v, current_adj, current_common, current_edges)

        # Local Search
        new_edges, new_adj, new_common, _ = local_search(current_edges, time_limit=start_time + max_time * 0.9)

        # Acceptance
        delta = len(new_edges) - len(current_edges)
        if delta > 0:
            # Accept improvement
            current_edges = new_edges
            current_adj = new_adj
            current_common = new_common
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = [[u, v] for u, v in current_edges]
        else:
            # Accept worsening with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current_edges = new_edges
                current_adj = new_adj
                current_common = new_common

        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final Polish: Try to add any remaining edges to the best solution
    if best_edges:
        final_adj = [0] * n
        final_common = [[0] * n for _ in range(n)]
        final_edges = []
        for u, v in best_edges:
            add_edge(u, v, final_adj, final_common, final_edges)

        # One more pass of greedy fill
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add(u, v, final_adj, final_common):
                add_edge(u, v, final_adj, final_common, final_edges)

        if len(final_edges) > best_count:
            best_count = len(final_edges)
            best_edges = [[u, v] for u, v in final_edges]

    if not best_edges:
        best_edges = []
        for i in range(5):
            base = i * 4
            for u in range(4):
                for v in range(u + 1, 4):
                    best_edges.append([base + u, base + v])

    if not verify_edges(best_edges):
        return [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
