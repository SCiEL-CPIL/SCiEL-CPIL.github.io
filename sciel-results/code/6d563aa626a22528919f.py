import time
import random
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 115

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    best_edges_list = []
    best_count = 0

    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)

        # Bitmask representation for fast C6 detection
        # adj_mask[i] is a bitmask of neighbors of i
        adj_mask = [0] * N
        edge_set = set()
        edges_list = []

        def add_edge(u, v):
            if (u, v) in edge_set or (v, u) in edge_set:
                return False
            edge_set.add((u, v))
            edges_list.append((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            return True

        def remove_edge(u, v):
            if (u, v) in edge_set:
                edge_set.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                if (u, v) in edges_list:
                    edges_list.remove((u, v))
            elif (v, u) in edge_set:
                edge_set.remove((v, u))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                if (v, u) in edges_list:
                    edges_list.remove((v, u))
            return True

        def has_c6_with_edge(u, v):
            """
            Check if adding edge (u,v) creates a C6.
            A C6 containing (u,v) corresponds to a path u-a-b-c-d-v of length 5.
            This is equivalent to: there exists a path of length 4 from u to some neighbor of v,
            not passing through v, and not revisiting u.
            
            We do BFS from u avoiding v, up to depth 4.
            Then check if any node at depth 4 is adjacent to v.
            """
            # Bitmask BFS
            visited = [0] * 5  # visited[d] = bitmask of nodes reached in exactly d steps from u (avoiding v)
            visited[0] = 1 << u
            
            for d in range(1, 5):
                mask = 0
                prev = visited[d-1]
                m = prev
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    # Neighbors of node, excluding v and excluding already visited nodes
                    # We need to be careful: simple path means no revisits
                    # But bitmask BFS without simple path constraint might over-estimate
                    # For C6, we need a simple path. Let's use a more careful approach.
                    # Actually, for a quick filter, let's just OR all neighbors except v
                    # Then at the end, we verify with a DFS if needed.
                    nb_mask = adj_mask[node] & ~(1 << v)
                    # Remove nodes in all previous layers to ensure simple path
                    for pd in range(d):
                        nb_mask &= ~visited[pd]
                    mask |= nb_mask
                visited[d] = mask

            # Check if any node at depth 4 is adjacent to v
            v_neigh = adj_mask[v] & ~(1 << u)  # Exclude u
            if (visited[4] & v_neigh) == 0:
                return False
            
            # Potential C6 found in BFS filter. Verify with DFS to extract actual path.
            # DFS from u to find path of length 4 to a neighbor of v
            # This is only called when the fast filter says "maybe", so it's rare
            def dfs_find_path(current, depth, path_set, path_list):
                if depth == 4:
                    if (adj_mask[current] & (1 << v)) != 0:
                        return True
                    return False
                nb_mask = adj_mask[current] & ~(1 << v)
                # Remove already visited
                for nb in path_list:
                    nb_mask &= ~(1 << nb)
                m = nb_mask
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    path_set.add(node)
                    path_list.append(node)
                    if dfs_find_path(node, depth + 1, path_set, path_list):
                        return True
                    path_list.pop()
                    path_set.remove(node)
                return False

            path_set = {u}
            path_list = [u]
            # Try to find a valid path
            # We need to try different branches. The DFS above tries all.
            # But we need to be careful about the starting point.
            # Actually, the BFS filter already confirmed existence of some path structure.
            # The DFS will find it if it exists.
            # However, the DFS as written might be slow if the graph is dense.
            # But we only call it when the filter passes, which should be rare in good graphs.
            
            # Optimization: If the graph is sparse, DFS is fast.
            # Let's just return True if the filter passes, to avoid the overhead of DFS in the critical path.
            # The risk is false positives, which would prevent adding a valid edge.
            # Given the goal is to maximize edges, false positives (missing a valid edge) are bad.
            # So we should verify.
            
            # Let's use a slightly faster DFS check
            # We only need to check if there is ANY path.
            # The DFS above is correct.
            return dfs_find_path(u, 0, path_set, path_list)

        def get_blocking_edge(u, v):
            """
            If (u,v) creates a C6, return one edge from that C6 to break.
            """
            if not has_c6_with_edge(u, v):
                return None
            
            # Find a specific C6 to break
            # Reuse the DFS logic but return the path
            def dfs_find_path_edges(current, depth, path_set, path_list):
                if depth == 4:
                    if (adj_mask[current] & (1 << v)) != 0:
                        # Found path: path_list + [v]
                        # Edges: (path_list[i], path_list[i+1]) for i in range(len-1), and (current, v)
                        # Return a list of edges in the cycle
                        # The cycle is path_list[0] - ... - path_list[4] - v - path_list[0]
                        # Edges: (u, a), (a, b), (b, c), (c, d), (d, v), (v, u)
                        # We want to remove one of the existing edges (not the hypothetical (u,v))
                        # So we can return (path_list[0], path_list[1])
                        return [(path_list[i], path_list[i+1]) for i in range(len(path_list)-1)]
                    return None
                
                nb_mask = adj_mask[current] & ~(1 << v)
                for nb in path_list:
                    nb_mask &= ~(1 << nb)
                m = nb_mask
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    path_set.add(node)
                    path_list.append(node)
                    res = dfs_find_path_edges(node, depth + 1, path_set, path_list)
                    if res:
                        return res
                    path_list.pop()
                    path_set.remove(node)
                return None

            path_set = {u}
            path_list = [u]
            res = dfs_find_path_edges(u, 0, path_set, path_list)
            if res:
                # Pick a random edge from the cycle to remove
                return random.choice(res)
            # Fallback: return None, meaning we can't find a specific edge
            # This shouldn't happen if has_c6_with_edge returned True
            return None

        # --- Initial Graph ---
        # Start with 8 disjoint K5s
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(base + a, base + b)

        local_best = len(edges_list)
        local_best_mask = list(adj_mask)

        # --- Greedy Phase ---
        def greedy_add():
            nonlocal local_best, local_best_mask
            added = 0
            while True:
                # Find a valid edge to add
                found = False
                # Shuffle pairs for randomness
                pairs = all_pairs[:]
                random.shuffle(pairs)
                for u, v in pairs:
                    if (u, v) in edge_set or (v, u) in edge_set:
                        continue
                    if not has_c6_with_edge(u, v):
                        add_edge(u, v)
                        added += 1
                        found = True
                        if len(edges_list) > local_best:
                            local_best = len(edges_list)
                            local_best_mask = list(adj_mask)
                        break
                if not found:
                    break
            return added

        # Local search loop
        while time.time() < deadline:
            added = greedy_add()
            if added == 0:
                # Stuck
                # Try a kick: remove an edge
                if edges_list:
                    # Choose an edge to remove
                    # Heuristic: remove an edge that is in many triangles or high-degree nodes?
                    # For now, random or from a blocking C6
                    # Try to find a blocking C6 for a random missing edge
                    missing = [(u, v) for u, v in all_pairs if (u, v) not in edge_set and (v, u) not in edge_set]
                    if missing:
                        u, v = random.choice(missing)
                        block_edge = get_blocking_edge(u, v)
                        if block_edge:
                            remove_edge(block_edge[0], block_edge[1])
                        else:
                            # Random removal
                            e = random.choice(edges_list)
                            remove_edge(e[0], e[1])
                    else:
                        break
                else:
                    break

        # Update global best
        if local_best > best_count:
            best_count = local_best
            best_edges_list = []
            for u in range(N):
                m = local_best_mask[u]
                while m:
                    low = m & (-m)
                    v = low.bit_length() - 1
                    m ^= low
                    if u < v:
                        best_edges_list.append([u, v])

        return local_best

    # Multi-restart
    seeds = [42, 123, 7, 999, 31415]
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        run_once(s)

    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
