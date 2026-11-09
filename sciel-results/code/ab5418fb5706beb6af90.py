import time

def construct():
    t_start = time.time()
    deadline = t_start + 85.0

    n = 21
    edges = [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]

    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
        adj[a].add(b)
        adj[b].add(a)

    edges_list = sorted(edge_set)
    edge_indices = {e: i for i, e in enumerate(edges_list)}
    
    # Precompute edge index matrix for O(1) lookup
    edge_idx_mat = [[-1] * n for _ in range(n)]
    for i, (u, v) in enumerate(edges_list):
        edge_idx_mat[u][v] = i
        edge_idx_mat[v][u] = i

    vertex_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges_list):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)

    def find_hamiltonian_cycle(available_mask, start):
        path = [start]
        used_mask = 0

        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                idx = edge_idx_mat[last][start]
                if idx != -1 and (available_mask & (1 << idx)) and not (used_mask & (1 << idx)):
                    return list(path)
                return None

            last = path[-1]
            candidates = []
            for nb in adj[last]:
                if nb in path:
                    continue
                idx = edge_idx_mat[last][nb]
                if idx == -1:
                    continue
                if not (available_mask & (1 << idx)):
                    continue
                if used_mask & (1 << idx):
                    continue
                
                # Heuristic: degree of nb in remaining available graph
                deg = 0
                for edge_idx in vertex_edges[nb]:
                    if (available_mask & (1 << edge_idx)) and not (used_mask & (1 << edge_idx)):
                        deg += 1
                candidates.append((deg, nb))
            
            candidates.sort()
            
            for deg, nb in candidates:
                idx = edge_idx_mat[last][nb]
                new_used_mask = used_mask | (1 << idx)
                path.append(nb)
                
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
            
            return None

        return backtrack()

    cycles = []
    available_mask = (1 << len(edges_list)) - 1

    starts = [0, 1, 2, 20, 19, 18]
    
    for _ in range(3):
        found = False
        for start in starts:
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(available_mask, start)
            if cycle is not None:
                cycles.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    idx = edge_idx_mat[u][v]
                    available_mask &= ~(1 << idx)
                found = True
                break
        if not found:
            break

    if len(cycles) == 0:
        return []

    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    return result
