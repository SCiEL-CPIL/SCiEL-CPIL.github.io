"""Cochromatic partitions of a pinned random graph – improved branch & bound."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12],
                         [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7],
                         [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12],
                         [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10],
                         [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12],
                         [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True

    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_indep[mask] = is_indep[rest] and ((adj[v] & rest) == 0)

    is_valid = [is_clique[m] or is_indep[m] for m in range(size)]

    # Precompute popcount
    pc = [0] * size
    for i in range(1, size):
        pc[i] = pc[i >> 1] + (i & 1)

    # Maximum clique / independent set sizes
    max_omega = max(pc[m] for m in range(1, size) if is_clique[m])
    max_alpha = max(pc[m] for m in range(1, size) if is_indep[m])
    max_part = max(max_omega, max_alpha)

    # Greedy initial solution
    def greedy(order):
        parts = []
        unassigned = (1 << n) - 1
        for v in order:
            v_bit = 1 << v
            if not (unassigned & v_bit):
                continue
            placed = False
            for i in range(len(parts)):
                if is_valid[parts[i] | v_bit]:
                    parts[i] |= v_bit
                    placed = True
                    break
            if not placed:
                parts.append(v_bit)
            unassigned ^= v_bit
        return parts

    orders = [
        list(range(n)),
        sorted(range(n), key=lambda v: -pc[adj[v]]),
        sorted(range(n), key=lambda v: pc[adj[v]]),
        sorted(range(n), key=lambda v: -pc[~adj[v] & ((1 << n) - 1)]),
    ]
    best_parts = min((greedy(o) for o in orders), key=len)
    best_count = len(best_parts)

    # Branch and bound
    start_time = time.time()
    deadline = start_time + 58.0

    def search(unassigned, current_parts, part_count):
        nonlocal best_parts, best_count

        if time.time() > deadline:
            return False

        if not unassigned:
            if part_count < best_count:
                best_count = part_count
                best_parts = current_parts[:]
            return True

        if part_count >= best_count:
            return False

        # Lower bound: remaining vertices need at least ceil(rem / max_part) more parts
        rem = pc[unassigned]
        if part_count + (rem + max_part - 1) // max_part >= best_count:
            return False

        # Try to fit all remaining into existing parts (optimistic check)
        # For each unassigned vertex, check if it can join some existing part
        # This is a necessary condition for not needing new parts
        u = unassigned
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            can_place = False
            for p in current_parts:
                if is_valid[p | lsb]:
                    can_place = True
                    break
            if not can_place:
                # This vertex needs a new part
                if part_count + 1 >= best_count:
                    return False
                break
            u ^= lsb

        # Select most constrained vertex
        best_v = -1
        min_placements = n + 1
        max_deg = -1
        u = unassigned
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            placements = 0
            for p in current_parts:
                if is_valid[p | lsb]:
                    placements += 1
            deg = pc[adj[v] & unassigned]
            if placements < min_placements or (placements == min_placements and deg > max_deg):
                min_placements = placements
                max_deg = deg
                best_v = v
                if placements == 0:
                    break
            u ^= lsb

        v = best_v
        v_bit = 1 << v
        new_unassigned = unassigned ^ v_bit

        # Try adding to existing parts
        indices = sorted(range(len(current_parts)), key=lambda i: pc[current_parts[i]])
        for i in indices:
            p = current_parts[i]
            if is_valid[p | v_bit]:
                current_parts[i] = p | v_bit
                if search(new_unassigned, current_parts, part_count):
                    current_parts[i] = p
                    return True
                current_parts[i] = p

        # Try new part
        current_parts.append(v_bit)
        if search(new_unassigned, current_parts, part_count + 1):
            current_parts.pop()
            return True
        current_parts.pop()

        return False

    search((1 << n) - 1, [], 0)

    # Convert to list of lists
    result = []
    for mask in best_parts:
        part_vertices = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        result.append(part_vertices)

    return result
# EVOLVE-BLOCK-END
