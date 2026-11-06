import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before the 60s limit

    n = 256
    all_mask = (1 << n) - 1

    # Precompute conflict masks
    # conflict[i] is a bitmask of indices j such that (i+1) and (j+1) conflict
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i

    # Complementary adjacency (H): edge (u,v) in H iff NOT conflict
    adj = [0] * n
    for i in range(n):
        adj[i] = (~conflict[i]) & all_mask
        adj[i] &= ~(1 << i)

    # Degree in H
    degrees = [adj[i].bit_count() for i in range(n)]

    # Order vertices by degree descending
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    rank = [0] * n
    for idx, v in enumerate(order):
        rank[v] = idx

    # Greedy coloring bound for MIS on candidate set P
    # Colors vertices such that no two adjacent in conflict graph share same color
    # Max color is upper bound on MIS size
    def color_bound(P):
        if P == 0:
            return 0
        verts = []
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            verts.append(i)
            p ^= lsb

        # Sort by rank (degree-based)
        verts.sort(key=lambda x: rank[x])

        colors = [-1] * n
        max_color = 0

        for v in verts:
            used = set()
            neigh = conflict[v] & P
            np_ = neigh
            while np_:
                lsb = np_ & -np_
                u = lsb.bit_length() - 1
                if colors[u] >= 0:
                    used.add(colors[u])
                np_ ^= lsb

            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c

        return max_color

    best_size = 0
    best_mask = 0
    found_cliques = []

    def bbmc(P, current):
        nonlocal best_size, best_mask

        if time.time() - start_time > timeout:
            return

        if P == 0:
            if len(current) > best_size:
                best_size = len(current)
                mask = 0
                for v in current:
                    mask |= (1 << v)
                best_mask = mask
                if len(current) >= 100:
                    found_cliques.append(list(current))
            return

        bound = color_bound(P)
        if len(current) + bound <= best_size:
            return

        # Pick vertex with lowest rank (highest degree)
        v = -1
        min_rank = n + 1
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            if rank[i] < min_rank:
                min_rank = rank[i]
                v = i
            p ^= lsb

        if v == -1:
            return

        v_bit = 1 << v

        # Include v
        new_P = P & adj[v]
        current.append(v)
        bbmc(new_P, current)
        current.pop()

        # Exclude v
        new_P_exc = P & ~v_bit
        bbmc(new_P_exc, current)

    # Initial greedy solution
    greedy_mask = 0
    P = all_mask
    while P:
        # Pick vertex with highest degree in remaining P
        v = -1
        max_deg = -1
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            deg = (adj[i] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = i
            p ^= lsb
        if v == -1:
            break
        greedy_mask |= (1 << v)
        P &= adj[v]  # Keep only neighbors of v in H

    if greedy_mask:
        best_size = greedy_mask.bit_count()
        best_mask = greedy_mask

    # Run BBMC
    bbmc(all_mask, [])

    # Local search improvement on best found
    if best_mask:
        # Try to add vertices
        improved = True
        while improved:
            improved = False
            P = best_mask
            for i in range(n):
                if not ((best_mask >> i) & 1):
                    # Check if i can be added
                    if (conflict[i] & best_mask) == 0:
                        best_mask |= (1 << i)
                        best_size += 1
                        improved = True
                        break
        # Try to swap: remove one, add two
        for i in range(n):
            if not ((best_mask >> i) & 1):
                for j in range(n):
                    if ((best_mask >> j) & 1):
                        new_mask = (best_mask & ~(1 << j)) | (1 << i)
                        # Check validity
                        valid = True
                        for k in range(n):
                            if (new_mask >> k) & 1 and k != i and k != j:
                                if (conflict[i] >> k) & 1:
                                    valid = False
                                    break
                        if valid and new_mask.bit_count() > best_size:
                            best_mask = new_mask
                            best_size = best_mask.bit_count()
                            improved = True
                            break
                if improved:
                    break

    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)

    if not result:
        return [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a = res[i]
            b = res[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
