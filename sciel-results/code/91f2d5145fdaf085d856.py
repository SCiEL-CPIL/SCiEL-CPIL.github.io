<<<<<<< SEARCH
    violation = find_violation()
    if violation is None:
        return {'n': n, 'edges': edges_20}
=======
    violation = find_violation()
    if violation is None:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
>>>>>>> REPLACE
