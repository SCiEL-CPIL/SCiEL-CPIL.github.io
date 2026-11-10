import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n = 18
    all_vertices = list(range(n))
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def is_k4_free(adj_mask):
        """Check if graph with adjacency masks is K4-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] & (1 << v)):
                    continue
                # Check if adding edge (u,v) would have created K4
                # Actually, check if there exist a,b such that {u,v,a,b} is K4
                # a and b must both connect to u and v, and a-b must be edge
                common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                # Check if any pair in common is connected
                temp = common
                while temp:
                    lsb = temp & (-temp)
                    a = lsb.bit_length() - 1
                    if adj_mask[a] & (common ^ lsb):
                        return False
                    temp ^= lsb
        return True

    def max_triangle_free_size(adj_mask):
        """Find the largest triangle-free induced subgraph using branch and bound."""
        # Precompute triangles as bitmasks
        triangles = []
        for a, b, c in combinations(range(n), 3):
            if (adj_mask[a] & (1 << b)) and (adj_mask[a] & (1 << c)) and (adj_mask[b] & (1 << c)):
                triangles.append((1 << a) | (1 << b) | (1 << c))

        if not triangles:
            return n

        best_size = [0]
        
        def search(idx, current_mask, current_size):
            if time.time() > deadline - 3:
                return
            if idx == n:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                return

            # Upper bound: current_size + (n - idx)
            if current_size + (n - idx) <= best_size[0]:
                return

            # Option 1: don't include vertex idx
            search(idx + 1, current_mask, current_size)

            # Option 2: include vertex idx if it doesn't create a triangle
            # Adding idx creates a triangle iff there exist u, v in current_mask
            # such that adj[u][v] and adj[u][idx] and adj[v][idx]
            # i.e., (adj_mask[idx] & current_mask) contains an edge
            new_neighbors = adj_mask[idx] & current_mask
            # Check if new_neighbors has any edge
            has_edge = False
            temp = new_neighbors
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if adj_mask[v] & (new_neighbors ^ lsb):
                    has_edge = True
                    break
                temp ^= lsb

            if not has_edge:
                search(idx + 1, current_mask | (1 << idx), current_size + 1)

        search(0, 0, 0)
        return best_size[0]

    def generate_random_k4_free():
        """Generate a random K4-free graph."""
        adj_mask = [0] * n
        edges = []
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)

        for u, v in shuffled_edges:
            # Check if adding (u,v) creates a K4
            # A K4 with edge (u,v) needs two other vertices a,b both connected to u and v, with a-b edge
            common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
            can_add = True
            temp = common
            while temp:
                lsb = temp & (-temp)
                a = lsb.bit_length() - 1
                if adj_mask[a] & (common ^ lsb):
                    can_add = False
                    break
                temp ^= lsb

            if can_add:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append((u, v))

        return edges, adj_mask

    # Try multiple random initializations and local search
    best_score = 12  # upper bound from K6,6,6
    best_edges = None
    best_adj = None

    # Generate several random K4-free graphs
    num_initial = 8
    candidates = []

    for i in range(num_initial):
        if time.time() > deadline - 30:
            break
        edges, adj_mask = generate_random_k4_free()
        score = max_triangle_free_size(adj_mask)
        candidates.append((score, edges, adj_mask))
        if score < best_score:
            best_score = score
            best_edges = edges
            best_adj = adj_mask

    # Also try the tripartite graph as a baseline
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    tripartite_edges = [[a, b] for i in range(3) for j in range(i+1, 3) for a in parts[i] for b in parts[j]]
    tripartite_adj = [0] * n
    for u, v in tripartite_edges:
        tripartite_adj[u] |= (1 << v)
        tripartite_adj[v] |= (1 << u)
    candidates.append((12, tripartite_edges, tripartite_adj))

    if best_edges is None:
        best_edges = tripartite_edges
        best_adj = tripartite_adj
        best_score = 12

    # Local search: try adding edges to reduce max triangle-free set
    if time.time() < deadline - 15:
        current_adj = list(best_adj)
        current_edges = list(best_edges)
        current_score = best_score

        # Generate all non-edges
        non_edges = []
        for u, v in all_edges:
            if not (current_adj[u] & (1 << v)):
                non_edges.append((u, v))

        random.shuffle(non_edges)

        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for u, v in non_edges:
                if time.time() > deadline - 10:
                    break
                if current_adj[u] & (1 << v):
                    continue

                # Check if adding (u,v) creates a K4
                common = current_adj[u] & current_adj[v] & ~(1 << u) & ~(1 << v)
                can_add = True
                temp = common
                while temp:
                    lsb = temp & (-temp)
                    a = lsb.bit_length() - 1
                    if current_adj[a] & (common ^ lsb):
                        can_add = False
                        break
                    temp ^= lsb

                if not can_add:
                    continue

                # Try adding this edge
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append((u, v))

                new_score = max_triangle_free_size(current_adj)

                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = list(current_edges)
                        best_adj = list(current_adj)
                else:
                    # Revert
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    current_edges.pop()

    # Return the best edges found
    if best_edges is None:
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        best_edges = [[a, b] for i in range(3) for j in range(i+1, 3) for a in parts[i] for b in parts[j]]

    return [[u, v] for u, v in best_edges]
