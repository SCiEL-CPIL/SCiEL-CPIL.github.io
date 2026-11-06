import random
import time

def construct():
    deadline = time.time() + 140
    n = 16

    # Start with K_{8,8} + one intra-part edge {0,1}
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((min(u, v), max(u, v)))
    edges.add((0, 1))

    def compute_score(edges_set):
        deg = [0] * n
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
        adj = [[False] * n for _ in range(n)]
        for u, v in edges_set:
            adj[u][v] = True
            adj[v][u] = True
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = compute_score(edges)
    best_score = current_score
    best_edges = set(edges)

    random.seed(42)

    def build_non_edges(e_set):
        ne = set()
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in e_set:
                    ne.add((i, j))
        return ne

    non_edges = build_non_edges(edges)

    # Main local search loop
    while time.time() < deadline - 15:
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True

        # Collect all triangles with degree sums
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            triangles.append((s, (i, j, k)))

        if not triangles:
            break

        triangles.sort(reverse=True)
        worst_sum, worst_tri = triangles[0]

        # Edges in the worst triangle, ordered by degree of endpoints (highest first)
        tri_edges = [(worst_tri[0], worst_tri[1]),
                     (worst_tri[0], worst_tri[2]),
                     (worst_tri[1], worst_tri[2])]
        # Sort by max degree of endpoints descending
        tri_edges.sort(key=lambda e: -(max(deg[e[0]], deg[e[1]])))

        improved = False
        for te in tri_edges:
            if time.time() >= deadline - 15:
                break
            candidates = list(non_edges)
            random.shuffle(candidates)
            for ne in candidates[:200]:
                if time.time() >= deadline - 15:
                    break
                new_edges = set(edges)
                new_edges.discard(te)
                new_edges.add(ne)
                if len(new_edges) != 65:
                    continue
                new_score = compute_score(new_edges)
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(edges)
                    non_edges = build_non_edges(edges)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Random edge swaps to escape local optima
            for _ in range(50):
                if time.time() >= deadline - 15:
                    break
                e = random.choice(list(edges))
                ne = random.choice(list(non_edges))
                new_edges = set(edges)
                new_edges.discard(e)
                new_edges.add(ne)
                if len(new_edges) != 65:
                    continue
                new_score = compute_score(new_edges)
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(edges)
                    non_edges = build_non_edges(edges)
                    break

    # Final exhaustive polish: try all single edge swaps from best solution
    if time.time() < deadline - 3:
        all_edges = list(best_edges)
        all_non_edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                         if (i, j) not in best_edges]
        for e in all_edges:
            if time.time() >= deadline - 3:
                break
            for ne in all_non_edges:
                if time.time() >= deadline - 3:
                    break
                new_edges = set(best_edges)
                new_edges.discard(e)
                new_edges.add(ne)
                if len(new_edges) != 65:
                    continue
                new_score = compute_score(new_edges)
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    break

    edge_list = sorted(best_edges)
    return {'n': n, 'edges': [[u, v] for u, v in edge_list]}
