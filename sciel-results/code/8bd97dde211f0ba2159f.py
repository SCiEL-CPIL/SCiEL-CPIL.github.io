import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 35s safety margin

    # Known good: two K_{3,3} -> {4,6} -> count=2
    def make_k33():
        edges = []
        for offset in (0, 6):
            part_a = [offset, offset+1, offset+2]
            part_b = [offset+3, offset+4, offset+5]
            for u in part_a:
                for v in part_b:
                    edges.append([u, v])
        return edges

    best_edges = make_k33()
    best_count = 2

    # Fast cycle-length counting using bitmasks and early termination
    def count_distinct_cycle_lengths(edges):
        n = 12
        adj_mask = [0] * n
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_list[u].append(v)
            adj_list[v].append(u)

        # Track which cycle lengths we've found
        found_lengths = [False] * (n + 1)  # index = length
        found_count = 0

        def dfs(current, start, path_len, visited_mask):
            nonlocal found_count
            for neighbor in adj_list[current]:
                if neighbor == start:
                    if path_len >= 3 and not found_lengths[path_len]:
                        found_lengths[path_len] = True
                        found_count += 1
                        if found_count == 1:
                            # Can't do better than 1, but we need to verify
                            # Actually we want to know ALL lengths, so keep going
                            pass
                elif neighbor > start and not (visited_mask & (1 << neighbor)):
                    dfs(neighbor, start, path_len + 1, visited_mask | (1 << neighbor))

        for start in range(n):
            visited_mask = 1 << start
            dfs(start, start, 1, visited_mask)

        return found_count

    # Local search from the known good graph
    random.seed(42)

    # Convert best_edges to a set for easy modification
    current_edges = set(tuple(sorted(e)) for e in best_edges)

    # Partitions for bipartite graph
    part_a = set(range(6))
    part_b = set(range(6, 12))

    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.9999
    min_temperature = 0.01

    for attempt in range(500000):
        if time.time() > deadline:
            break

        # Pick two edges to swap
        edge_list = list(current_edges)
        if len(edge_list) < 2:
            break

        e1, e2 = random.sample(edge_list, 2)

        # They must be disjoint
        v1, v2 = e1
        v3, v4 = e2
        if len({v1, v2, v3, v4}) < 4:
            continue

        # Identify a-part and b-part vertices
        if v1 in part_a:
            a1, b1 = v1, v2
        else:
            a1, b1 = v2, v1

        if v3 in part_a:
            a2, b2 = v3, v4
        else:
            a2, b2 = v4, v3

        # New edges
        new_e1 = (min(a1, b2), max(a1, b2))
        new_e2 = (min(a2, b1), max(a2, b1))

        # Check if new edges already exist
        if new_e1 in current_edges or new_e2 in current_edges:
            continue

        # Apply swap
        current_edges.remove(e1)
        current_edges.remove(e2)
        current_edges.add(new_e1)
        current_edges.add(new_e2)

        edge_list = [[u, v] for u, v in current_edges]

        # Count distinct cycle lengths
        cnt = count_distinct_cycle_lengths(edge_list)

        if cnt < best_count:
            best_count = cnt
            best_edges = edge_list
            if best_count == 1:
                break
        else:
            # Simulated annealing: accept worse solutions with probability exp(-delta/T)
            delta = cnt - best_count
            if delta > 0 and temperature > min_temperature:
                import math
                if random.random() < math.exp(-delta / temperature):
                    # Accept the worse solution to escape local optima
                    temperature *= cooling_rate
                    continue
                else:
                    # Revert swap
                    current_edges.remove(new_e1)
                    current_edges.remove(new_e2)
                    current_edges.add(e1)
                    current_edges.add(e2)
                    temperature *= cooling_rate
            else:
                # Revert swap
                current_edges.remove(new_e1)
                current_edges.remove(new_e2)
                current_edges.add(e1)
                current_edges.add(e2)
                temperature *= cooling_rate

    # Ensure types are correct
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])

    return result
