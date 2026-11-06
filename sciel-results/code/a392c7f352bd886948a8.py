"""Combined program: greedy C4-free construction with local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_c4_free(adj, n):
        """Check C4-freeness: no two adjacent vertices share 2+ common neighbors."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    # Count common neighbors of i and j
                    common = np.bitwise_and(adj[i], adj[j]).sum()
                    if common >= 2:
                        return False
        return True

    def min_degree(adj, n):
        return int(min(adj[i].sum() for i in range(n)))

    def try_construct(n, attempts=30):
        target = n - 39
        if target < 1:
            target = 1
        best_adj = None
        best_md = -1

        for attempt in range(attempts):
            if elapsed() > deadline - 3:
                break

            # Initialize: start with a random graph at moderate density
            p = target / n * 1.2
            adj = (np.random.rand(n, n) < p).astype(np.int8)
            np.fill_diagonal(adj, 0)
            adj = np.triu(adj, 1)
            adj = adj + adj.T

            # Phase 1: Remove edges that create C4s
            changed = True
            max_iter = 200
            for _ in range(max_iter):
                if not changed:
                    break
                changed = False
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            common = np.bitwise_and(adj[i], adj[j]).sum()
                            if common >= 2:
                                # Remove edge (i,j) to break C4
                                adj[i][j] = 0
                                adj[j][i] = 0
                                changed = True
                if elapsed() > deadline - 3:
                    break

            if not check_c4_free(adj, n):
                continue

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj.copy()

            # Phase 2: Local search - try to add edges to improve min degree
            improved = True
            max_local = 50
            for _ in range(max_local):
                if not improved:
                    break
                improved = False
                if elapsed() > deadline - 3:
                    break
                # Find vertices with minimum degree
                degs = adj.sum(axis=1)
                min_deg = int(degs.min())
                if min_deg >= target:
                    break
                low_verts = np.where(degs <= min_deg + 1)[0]
                random.shuffle(low_verts)
                for u in low_verts:
                    if elapsed() > deadline - 3:
                        break
                    # Try adding edges from u to non-neighbors
                    non_neighbors = np.where(adj[u] == 0)[0]
                    non_neighbors = non_neighbors[non_neighbors != u]
                    random.shuffle(non_neighbors)
                    for v in non_neighbors:
                        # Check if adding (u,v) creates a C4
                        common = np.bitwise_and(adj[u], adj[v]).sum()
                        if common < 2:
                            adj[u][v] = 1
                            adj[v][u] = 1
                            improved = True
                            if degs[u] >= target and degs[v] >= target:
                                break

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj.copy()

            if md >= target:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return n, edges

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i][j]:
                        edges.append([i, j])
            return n, edges
        return None

    # Try from largest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
