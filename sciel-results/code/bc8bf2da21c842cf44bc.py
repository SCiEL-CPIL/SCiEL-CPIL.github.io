import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build the complement graph H where H[u] has bit v set iff
    #    u != v and (u-v) mod m is NOT in forbidden.
    #    We want a Maximum Clique in H (which is a Maximum Independent Set in G).
    #
    #    Pre-compute degrees in H.
    
    comp_adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v and ((u - v) % m) not in forbidden:
                mask |= (1 << v)
        comp_adj[u] = mask

    # degrees in H
    deg = [0] * m
    for u in range(m):
        deg[u] = bin(comp_adj[u]).count('1')

    deadline = time.time() + 590.0  # safety margin before 630s wall clock

    def time_exceeded():
        return time.time() > deadline

    # Tomita maximum clique algorithm with greedy coloring bound.
    best_clique = []
    best_size = 0

    def expand(clique, pool_mask):
        """
        clique: tuple/list of vertices already in the current clique
        pool_mask: bitmask of candidate vertices (adjacent to all in clique in H)
        """
        nonlocal best_clique, best_size

        if time_exceeded():
            return

        # If pool is empty, record best
        if pool_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return

        # Greedy coloring of the subgraph induced by pool_mask
        # Color each vertex with the smallest color not used by its colored neighbors
        # Process vertices in order of decreasing degree in H (within pool)
        # for tighter bound.

        # Extract candidate vertices from pool_mask
        cand = []
        tmp = pool_mask
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            cand.append(v)
            tmp ^= lsb

        # Sort by degree in H descending (pre-computed global degree is a good proxy)
        cand.sort(key=lambda x: deg[x], reverse=True)

        color_of = [0] * m  # reset each call (m=205, cheap)
        max_color = 0
        for v in cand:
            # Find colors used by neighbors of v in H that are in pool and already colored
            neigh = comp_adj[v] & pool_mask
            used = set()
            nm = neigh
            while nm:
                lsb = nm & (-nm)
                u = lsb.bit_length() - 1
                if color_of[u] > 0:
                    used.add(color_of[u])
                nm ^= lsb
            c = 1
            while c in used:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c

        # Branch: iterate vertices in decreasing color order
        # Re-sort by color descending (stable: ties broken by degree descending from above)
        cand_with_color = [(v, color_of[v]) for v in cand]
        cand_with_color.sort(key=lambda x: x[1], reverse=True)

        current_pool = pool_mask

        for v, c in cand_with_color:
            if time_exceeded():
                return

            # Bound: even adding this vertex and all higher-color vertices
            # cannot beat best_size
            if len(clique) + c <= best_size:
                break

            # Check v is still in current_pool
            if not (current_pool & (1 << v)):
                continue

            # Recurse with v added
            new_pool = current_pool & comp_adj[v]
            new_clique = tuple(clique) + (v,)
            expand(new_clique, new_pool)

            # Remove v from pool for subsequent iterations
            current_pool &= ~(1 << v)

    # Initial call
    initial_pool = (1 << m) - 1
    expand((), initial_pool)

    if not best_clique:
        return np.array([0], dtype=np.int64)

    return np.array(sorted(best_clique), dtype=np.int64)
