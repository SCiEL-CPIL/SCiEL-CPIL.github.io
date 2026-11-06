import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before the 90s limit

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
    # Optimized: use pre-sorted order, avoid repeated list construction
    def color_bound(P):
        if P == 0:
            return 0
        # Use the pre-sorted order, filter to only vertices in P
        verts = [v for v in order if (P >> v) & 1]

        if not verts:
            return 0

        colors = [-1] * n
        max_color = 0

        for v in verts:
            used = 0  # Use bitmask for used colors (faster than set for small color counts)
            neigh = conflict[v] & P
            np_ = neigh
            while np_:
                lsb = np_ & -np_
                u = lsb.bit_length() - 1
                cu = colors[u]
                if cu >= 0:
                    used |= (1 << cu)
                np_ ^= lsb

            c = 0
            while (used >> c) & 1:
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

    # Additional local search passes after BBMC
    if best_mask:
        # Multiple rounds of add-then-swap local search
        for _round in range(3):
            if time.time() - start_time > timeout:
                break
            improved = True
            while improved:
                improved = False
                # Try to add any vertex that doesn't conflict
                for i in range(n):
                    if not ((best_mask >> i) & 1):
                        if (conflict[i] & best_mask) == 0:
                            best_mask |= (1 << i)
                            best_size += 1
                            improved = True
                            break
                if time.time() - start_time > timeout:
                    break
            # Try swaps: remove one, add one (net zero or positive)
            for i in range(n):
                if not ((best_mask >> i) & 1):
                    # Check if i conflicts with any in best_mask
                    if (conflict[i] & best_mask) == 0:
                        continue  # Already handled by add
                    # Find a vertex j in best_mask that conflicts with i,
                    # and check if removing j and adding i still works
                    conf_with_i = conflict[i] & best_mask
                    cp = conf_with_i
                    while cp:
                        lsb = cp & -cp
                        j = lsb.bit_length() - 1
                        new_mask = (best_mask & ~(1 << j)) | (1 << i)
                        # Verify: i must not conflict with any other in new_mask
                        rest = new_mask & ~(1 << i)
                        if (conflict[i] & rest) == 0:
                            best_mask = new_mask
                            improved = True
                            break
                        cp ^= lsb
                    if improved or time.time() - start_time > timeout:
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
