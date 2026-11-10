import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def greedy_packing_partition(adj, n, deadline):
        """
        Greedy edge-coloring on the line graph: assign each edge to an existing
        clique if it fits (all vertices in the clique are adjacent to both endpoints),
        otherwise create a new clique. Process edges in order of "difficulty"
        (edges with fewer common neighbors first) to pack harder edges early.
        """
        # Collect all edges
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask ^= lsb

        if not edges:
            return []

        # Sort edges by number of common neighbors (ascending) - harder edges first
        # An edge (u,v) can be added to a clique C iff for all w in C, w is adjacent to both u and v
        # i.e., w is in adj[u] & adj[v]
        def edge_difficulty(e):
            u, v = e
            return (adj[u] & adj[v]).bit_count()

        edges.sort(key=edge_difficulty)

        # cliques: list of (frozen_set_of_vertices, bitmask)
        cliques = []  # list of bitmasks

        for (u, v) in edges:
            if time.time() > deadline:
                break
            u_mask = 1 << u
            v_mask = 1 << v
            # Common neighbors of u and v (excluding u and v themselves)
            common = adj[u] & adj[v] & ~(u_mask | v_mask)

            assigned = False
            # Try to fit into an existing clique
            # Prefer cliques that are "tighter" (smaller) to save larger ones?
            # Or prefer cliques where the edge fits with minimal conflict?
            # Strategy: try cliques in order of size (smaller first) to keep larger cliques flexible
            # Actually, for packing, we want to use up cliques that can't accept many more edges.
            # Let's try: for each existing clique, check if u and v can be added.
            # To add u,v to clique C: every vertex in C must be adjacent to both u and v.
            # i.e., C & ~(common | u_mask | v_mask) == 0  ... wait
            # Every w in C must satisfy: w in adj[u] AND w in adj[v]
            # i.e., C & ~(adj[u] & adj[v]) == 0
            # But u and v themselves: if u is already in C, then we only need v to be adjacent to all in C.
            # Let's handle cases:
            # Case 1: u in C, v not in C: need C \ {u} subset of adj[v]
            # Case 2: v in C, u not in C: need C \ {v} subset of adj[u]
            # Case 3: neither in C: need C subset of adj[u] & adj[v]
            # Case 4: both in C: edge already covered, shouldn't happen

            # For simplicity and speed, let's just check: can we add both u and v to C?
            # New clique would be C | {u, v}. For this to be a clique, every pair in C|{u,v} must be edges.
            # Pairs within C are already edges. Pairs (u,w) for w in C: need w in adj[u]. Pairs (v,w) for w in C: need w in adj[v]. Pair (u,v): already an edge.
            # So condition: C & ~adj[u] == 0 AND C & ~adj[v] == 0
            # i.e., (C & ~(adj[u] & adj[v])) == 0

            best_clique_idx = -1
            best_score = -1

            for idx, c_mask in enumerate(cliques):
                # Check if adding u and v to c_mask keeps it a clique
                # Need: c_mask & ~(adj[u] & adj[v]) == 0
                # But also u and v themselves might already be in c_mask
                # If u is in c_mask, we don't need to add it, just check v against rest
                # Let's handle uniformly: the new set is c_mask | u_mask | v_mask
                # For this to be a clique, every vertex in (c_mask | u_mask | v_mask) must be adjacent to every other.
                # Since c_mask is a clique, we only need:
                # - For all w in c_mask: w adjacent to u AND w adjacent to v (if u,v not already in c_mask)
                # - u adjacent to v (guaranteed)
                # If u in c_mask: need all w in c_mask adjacent to v
                # If v in c_mask: need all w in c_mask adjacent to u

                u_in = bool(c_mask & u_mask)
                v_in = bool(c_mask & v_mask)

                if u_in and v_in:
                    # Edge already in this clique, skip
                    continue

                if u_in:
                    # Need c_mask \ {u} subset of adj[v]
                    if (c_mask & ~u_mask) & ~adj[v] != 0:
                        continue
                    # Score: prefer smaller cliques to keep large ones flexible
                    score = -(c_mask.bit_count())
                elif v_in:
                    # Need c_mask \ {v} subset of adj[u]
                    if (c_mask & ~v_mask) & ~adj[u] != 0:
                        continue
                    score = -(c_mask.bit_count())
                else:
                    # Need c_mask subset of adj[u] & adj[v]
                    if c_mask & ~(adj[u] & adj[v]) != 0:
                        continue
                    score = -(c_mask.bit_count())

                if score > best_score:
                    best_score = score
                    best_clique_idx = idx

            if best_clique_idx >= 0:
                cliques[best_clique_idx] |= u_mask | v_mask
                assigned = True

            if not assigned:
                cliques.append(u_mask | v_mask)

        # Convert bitmasks to vertex lists
        result = []
        for c_mask in cliques:
            verts = []
            tmp = c_mask
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                verts.append(v)
                tmp ^= lsb
            if len(verts) >= 2:
                result.append(verts)

        return result

    def validate_and_fix(cliques, adj, n):
        """
        Validate that every edge is in exactly one clique and each clique is a valid clique.
        If invalid, fall back to edge-by-edge 2-clique cover.
        """
        # Build set of covered edges
        covered = set()
        for clique in cliques:
            if len(clique) < 2:
                return False
            # Check clique validity
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v:
                        u, v = v, u
                    if not (adj[u] >> v) & 1:
                        return False
                    if (u, v) in covered:
                        return False
                    covered.add((u, v))

        # Check all edges are covered
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    if (u, v) not in covered:
                        return False
                mask ^= lsb

        return True

    def fallback_partition(adj, n):
        """Cover each edge as a 2-clique."""
        cliques = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    cliques.append([u, v])
                mask ^= lsb
        return cliques

    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin before 150s kill

    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)

    # Process instance 1
    mid_deadline = start_time + 70
    cliques_1 = greedy_packing_partition(adj_1, 32, mid_deadline)
    if not validate_and_fix(cliques_1, adj_1, 32):
        cliques_1 = fallback_partition(adj_1, 32)

    # Process instance 2
    cliques_2 = greedy_packing_partition(adj_2, 64, deadline)
    if not validate_and_fix(cliques_2, adj_2, 64):
        cliques_2 = fallback_partition(adj_2, 64)

    return [cliques_1, cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
