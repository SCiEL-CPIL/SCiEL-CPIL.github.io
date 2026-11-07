import time

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

# Precompute trees and adjacency
TREES = instance_trees()
N = 20
tree_edges = [set((min(u, v), max(u, v)) for u, v in t) for t in TREES]
tree_adj = [{} for _ in range(20)]
for k in range(20):
    for u, v in tree_edges[k]:
        tree_adj[k].setdefault(u, set()).add(v)
        tree_adj[k].setdefault(v, set()).add(u)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    used_edges = set()
    solution = [None] * 20

    def backtrack(k, current_mapping):
        if k == 20:
            return True

        if time.time() > deadline:
            return False

        if k == 0:
            solution[0] = [0]
            return backtrack(1, [0])

        if k == 1:
            # T2: edge (0,1)
            if (0, 1) not in used_edges:
                used_edges.add((0, 1))
                solution[1] = [0, 1]
                if backtrack(2, [0, 1]):
                    return True
                used_edges.remove((0, 1))
                solution[1] = None
            return False

        # Helper to get free neighbors of a host vertex
        def get_free_neighbors(host, mapping):
            free = []
            for w in range(N):
                if w == host:
                    continue
                if w in mapping:
                    continue
                e = (min(host, w), max(host, w))
                if e not in used_edges:
                    free.append(w)
            return free

        def assign(idx, mapping):
            if idx == k:
                # Check edges
                for u, v in tree_edges[k]:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        return False

                # Add edges
                for u, v in tree_edges[k]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))

                solution[k] = mapping.copy()

                if backtrack(k + 1, mapping):
                    return True

                # Undo
                for u, v in tree_edges[k]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.remove((min(hu, hv), max(hu, hv)))
                solution[k] = None
                return False

            # Pruning: Check if current vertex idx can be assigned to any host
            candidates = []
            for host in range(N):
                if host in mapping:
                    continue

                # Check if host has enough free neighbors for unassigned neighbors of idx
                unassigned_neighbors = []
                for neighbor in tree_adj[k].get(idx, set()):
                    if neighbor not in mapping:
                        unassigned_neighbors.append(neighbor)

                # Also need to check assigned neighbors
                # For each assigned neighbor, the edge must be free.
                valid = True
                for neighbor in tree_adj[k].get(idx, set()):
                    if neighbor in mapping:
                        h_neighbor = mapping[neighbor]
                        e = (min(host, h_neighbor), max(host, h_neighbor))
                        if e in used_edges:
                            valid = False
                            break

                if not valid:
                    continue

                free_neighbors = get_free_neighbors(host, mapping)
                # We need to pick distinct free neighbors for the unassigned neighbors
                if len(free_neighbors) < len(unassigned_neighbors):
                    continue

                candidates.append(host)

            for host in candidates:
                mapping[idx] = host
                if assign(idx + 1, mapping):
                    return True
                mapping[idx] = None

            return False

        # Try all hosts for vertex 0 of T_k
        for host0 in range(N):
            mapping = [None] * k
            mapping[0] = host0
            if assign(1, mapping):
                return True

        return False

    if backtrack(0, []):
        return solution
    else:
        return [None] * 20
