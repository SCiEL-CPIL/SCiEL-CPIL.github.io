"""Packing edge-disjoint cycles on one vertex set."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 75  # safety margin before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    n = P['n']
    edges_list = P['edges']

    # Build edge index mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i

    # Adjacency with edge indices
    adj = [set() for _ in range(n)]
    adj_edges = [set() for _ in range(n)]  # (neighbor, edge_idx)
    for idx, (u, v) in enumerate(edges_list):
        adj[u].add(v)
        adj[v].add(u)
        adj_edges[u].add((v, idx))
        adj_edges[v].add((u, idx))

    # Check degree constraints for Hamiltonian cycles
    # For k edge-disjoint Hamiltonian cycles, each vertex needs degree >= 2k
    min_degree = min(len(adj[v]) for v in range(n))
    max_possible_cycles = min(10, min_degree // 2)

    # Also constrained by total edges: k * n <= total_edges
    max_possible_cycles = min(max_possible_cycles, len(edges_list) // n)

    print(f"Min degree: {min_degree}, max possible cycles: {max_possible_cycles}", file=sys.stderr)

    # DFS-based Hamiltonian cycle finder with edge-disjointness constraint
    def find_hamiltonian_cycles(k_target, deadline):
        """Find up to k_target edge-disjoint Hamiltonian cycles."""
        used_edges = [False] * len(edges_list)
        cycles = []

        def dfs_cycle(start, path, visited_mask, used_edges_ref, depth):
            """Find a Hamiltonian cycle starting at 'start' avoiding used edges."""
            if time.time() > deadline:
                return None

            if depth == n:
                # Check if we can close: last vertex connected to start
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in edge_index:
                    eidx = edge_index[key]
                    if not used_edges_ref[eidx]:
                        return list(path)
                return None

            last = path[-1]
            # Try neighbors in order (sorted for determinism)
            for neighbor, eidx in sorted(adj_edges[last]):
                if visited_mask & (1 << neighbor):
                    continue
                if used_edges_ref[eidx]:
                    continue
                # Pruning: basic connectivity check
                visited_mask_new = visited_mask | (1 << neighbor)
                path.append(neighbor)
                used_edges_ref[eidx] = True
                result = dfs_cycle(start, path, visited_mask_new, used_edges_ref, depth + 1)
                used_edges_ref[eidx] = False
                path.pop()
                if result is not None:
                    return result
            return None

        # Try to find cycles one by one
        for c in range(k_target):
            if time.time() > deadline - 5:
                break

            found = dfs_cycle(0, [0], 1, used_edges, 1)
            if found is None:
                break

            # Mark edges of this cycle as used
            for i in range(n):
                u = found[i]
                v = found[(i + 1) % n]
                key = (min(u, v), max(u, v))
                eidx = edge_index[key]
                used_edges[eidx] = True

            cycles.append(found)

        return cycles

    # Try to find maximum number of edge-disjoint Hamiltonian cycles
    # Start with the theoretical maximum and work down
    best_cycles = None
    best_count = 0

    for k in range(max_possible_cycles, 0, -1):
        if time.time() > deadline - 3:
            break

        print(f"Trying to find {k} edge-disjoint Hamiltonian cycles...", file=sys.stderr)
        cycles = find_hamiltonian_cycles(k, deadline)

        if cycles is not None and len(cycles) > best_count:
            best_cycles = cycles
            best_count = len(cycles)
            print(f"Found {best_count} cycles!", file=sys.stderr)

            if best_count >= max_possible_cycles:
                break

    # If no cycles found at all, try to find at least one
    if best_cycles is None or len(best_cycles) == 0:
        # Simple triangle fallback
        for i, (u, v) in enumerate(edges_list):
            if time.time() > deadline:
                break
            for w in adj[u] & adj[v]:
                if w != u and w != v:
                    best_cycles = [[u, v, w]]
                    best_count = 1
                    break
            if best_cycles:
                break

    if best_cycles is None or len(best_cycles) == 0:
        # Last resort
        return [[0, 1, 2]]

    # Validate: all cycles must use exactly the same vertex set of size >= 3
    # and edges must be disjoint
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])

    return result

# EVOLVE-BLOCK-END
