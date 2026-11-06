"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks for fast intersection
        rem_mask = [0] * n
        for u, v in edges:
            rem_mask[u] |= (1 << v)
            rem_mask[v] |= (1 << u)

        remaining_count = len(edges)
        cliques = []

        # Helper to get list of neighbors from mask
        def mask_to_list(mask):
            res = []
            while mask:
                lsb = mask & (-mask)
                res.append(lsb.bit_length() - 1)
                mask ^= lsb
            return res

        def find_max_clique():
            """
            Finds the maximum clique using Tomita's branch and bound algorithm.
            Returns a list of vertices forming a clique.
            """
            if not any(rem_mask):
                return []
            
            best_clique = []
            best_size = 0
            
            P = [v for v in range(n) if rem_mask[v]]
            if not P:
                return []
            
            def expand(P_list, clique):
                nonlocal best_clique, best_size
                
                if not P_list:
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique[:]
                    return
                
                # Color P_list to get upper bounds
                colors = [0] * len(P_list)
                for i, v in enumerate(P_list):
                    used_colors = set()
                    for j in range(i):
                        u = P_list[j]
                        if (rem_mask[v] >> u) & 1:
                            used_colors.add(colors[j])
                    
                    c = 1
                    while c in used_colors:
                        c += 1
                    colors[i] = c
                
                # Sort vertices by color
                P_sorted = sorted(zip(colors, P_list), key=lambda x: x[0])
                
                # Process in reverse order (highest color first)
                for i in range(len(P_sorted) - 1, -1, -1):
                    c, v = P_sorted[i]
                    # Prune if current clique + max possible extension < best
                    if len(clique) + c < best_size:
                        break
                    
                    # New candidates are neighbors of v in the unprocessed part of P (indices < i)
                    new_P = [u for c_u, u in P_sorted[:i] if (rem_mask[v] >> u) & 1]
                    
                    expand(new_P, clique + [v])

            expand(P, [])
            
            return best_clique

        while remaining_count > 0:
            if time.time() > deadline - 10:
                break

            clique = find_max_clique()

            if not clique or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue

            # Remove all edges within clique from remaining
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i + 1:]:
                    if (rem_mask[u] >> v) & 1:
                        rem_mask[u] &= ~(1 << v)
                        rem_mask[v] &= ~(1 << u)
                        removed += 1

            if removed == 0:
                # Shouldn't happen, but fallback
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            break
                        mask ^= lsb
                    if remaining_count < len(edges):
                        break
                continue

            cliques.append(clique)
            remaining_count -= removed

        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = rem_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem_mask[u] &= ~(1 << v)
                    rem_mask[v] &= ~(1 << u)
                mask ^= lsb

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
# EVOLVE-BLOCK-END
