import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s limit

    def time_exceeded():
        return time.time() - start_time > deadline

    # Precompute quadratic residues for q=29
    q = 29
    squares_29 = set()
    for x in range(1, q):
        squares_29.add((x * x) % q)
    # squares_29 = {1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28}

    def build_paley_colors(n):
        """Build Paley coloring for prime n ≡ 1 (mod 4)."""
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)

    def check_k222_fast(n, colors):
        """
        Fast check for K_{2,2,2} in either color class using bitmasks.
        Returns (bool, list) where bool is whether K_{2,2,2} exists,
        and list is a witness (6-tuple) if found, else [].
        """
        # Build bitmask adjacency for each color
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1

        def has_k222(adj_color):
            """
            Check if adj_color contains K_{2,2,2}.
            Returns (True, witness) or (False, []).
            K_{2,2,2}: 6 vertices partitioned into 3 pairs (a,b), (c,d), (e,f)
            such that all cross-pair edges exist and no intra-pair edges exist.
            """
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_color[i] & (1 << j):
                        continue  # skip pairs with an edge

                    # Common neighbors of i and j
                    cn_mask = (adj_color[i] & adj_color[j]) & ~(1 << i) & ~(1 << j)

                    # Get list of common neighbors
                    cn_list = []
                    mask = cn_mask
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        cn_list.append(v)
                        mask ^= lsb

                    if len(cn_list) < 4:
                        continue

                    # Find pairs (k,l) in cn_list with no edge
                    no_edge_pairs = []
                    for a_idx in range(len(cn_list)):
                        k = cn_list[a_idx]
                        for b_idx in range(a_idx + 1, len(cn_list)):
                            l = cn_list[b_idx]
                            if not (adj_color[k] & (1 << l)):
                                no_edge_pairs.append((k, l))

                    for (k, l) in no_edge_pairs:
                        # Common neighbors of i, j, k, l
                        cn2_mask = (adj_color[i] & adj_color[j] & adj_color[k] & adj_color[l])
                        cn2_mask &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)

                        # Get list
                        cn2_list = []
                        mask2 = cn2_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            cn2_list.append(v2)
                            mask2 ^= lsb2

                        if len(cn2_list) < 2:
                            continue

                        for c_idx in range(len(cn2_list)):
                            m = cn2_list[c_idx]
                            for d_idx in range(c_idx + 1, len(cn2_list)):
                                o = cn2_list[d_idx]
                                if not (adj_color[m] & (1 << o)):
                                    return True, (i, j, k, l, m, o)
            return False, []

        res0, wit0 = has_k222(adj[0])
        if res0:
            return True, wit0
        res1, wit1 = has_k222(adj[1])
        if res1:
            return True, wit1
        return False, []

    # Verify Paley(29) is valid
    valid_29, _ = check_k222_fast(q, colors_29)

    if not valid_29:
        # Paley(29) is valid! Try to extend to n=30 via local search.
        if not time_exceeded():
            best_n = q
            best_colors = colors_29[:]

            import random
            random.seed(42)

            # Strategy: Try to extend to n=30 using local search.
            # Start with a random extension, then fix conflicts.
            
            max_attempts = 30
            found_30 = False
            for attempt in range(max_attempts):
                if time_exceeded():
                    break

                # Create initial coloring for n=30
                # Start with Paley(29) and randomly assign edges to vertex 30
                colors_30 = colors_29[:]
                # Edges involving vertex 30 are the last 29 in lexicographic order
                # Pairs: (0,30), (1,30), ..., (28,30)
                for i in range(29):
                    colors_30.append(random.randint(0, 1))

                # Local search: repeatedly find a K_{2,2,2} and flip an edge to break it
                # We use a "steepest descent" like approach: pick a random edge from the witness to flip.
                for iteration in range(2000):
                    if time_exceeded():
                        break

                    has_conflict, witness = check_k222_fast(30, colors_30)
                    if not has_conflict:
                        # Found valid n=30 coloring!
                        return {'n': 30, 'colors': colors_30}
                    
                    # We have a witness (i, j, k, l, m, o) forming K_{2,2,2}.
                    # We need to break it. We can flip any edge involved in the structure.
                    # The structure requires:
                    # - No edges: (i,j), (k,l), (m,o)
                    # - All other cross edges present.
                    # To break it, we can:
                    # 1. Add an edge to one of the "no edge" pairs (turns color 0->1 or 1->0 depending on which color class).
                    # 2. Remove an edge from one of the "cross" pairs.
                    
                    # The witness is from a specific color class.
                    # Let's determine which color class.
                    # Actually, check_k222_fast returns the witness from whichever color class found it first.
                    # We need to know the color of the edges to decide how to flip.
                    # Let's just pick a random edge from the 6 vertices and flip it.
                    # The 6 vertices are i, j, k, l, m, o.
                    # There are 15 pairs.
                    verts = [witness[0], witness[1], witness[2], witness[3], witness[4], witness[5]]
                    # Pick a random pair from these 6 vertices
                    pair_idx1 = random.randint(0, 5)
                    pair_idx2 = random.randint(0, 5)
                    if pair_idx1 == pair_idx2:
                        pair_idx2 = (pair_idx2 + 1) % 6
                    v1, v2 = verts[pair_idx1], verts[pair_idx2]
                    if v1 > v2:
                        v1, v2 = v2, v1
                    
                    # Find index of edge (v1, v2) in colors_30
                    # Index formula for edge (i, j) with i < j in n vertices:
                    # sum_{k=0}^{i-1} (n - 1 - k) + (j - i - 1)
                    # Or simply iterate. For n=30, 435 edges. Iteration is fast enough if done rarely,
                    # but we can precompute a lookup table.
                    
                    # Let's precompute a lookup table for edge indices for n=30
                    if not hasattr(construct, 'edge_lookup_30'):
                        lookup = {}
                        idx = 0
                        for a in range(30):
                            for b in range(a + 1, 30):
                                lookup[(a, b)] = idx
                                idx += 1
                        construct.edge_lookup_30 = lookup
                    
                    edge_idx = construct.edge_lookup_30[(v1, v2)]
                    colors_30[edge_idx] = 1 - colors_30[edge_idx]

                # If we exit the loop without finding a valid one, try next attempt
                # We could also keep the best state (fewest conflicts) but that's complex.
                # Simple random restart is often effective for small n.

            # If we couldn't find n=30, return the best valid coloring (which is n=29)
            return {'n': best_n, 'colors': best_colors}
        else:
            return {'n': q, 'colors': colors_29}
    else:
        # Paley(29) is NOT valid - this shouldn't happen, but handle it.
        # Try to repair Paley(29) via local search.
        import random
        random.seed(42)
        
        colors = colors_29[:]
        for iteration in range(5000):
            if time_exceeded():
                break
            
            has_conflict, witness = check_k222_fast(29, colors)
            if not has_conflict:
                return {'n': 29, 'colors': colors}
            
            verts = [witness[0], witness[1], witness[2], witness[3], witness[4], witness[5]]
            pair_idx1 = random.randint(0, 5)
            pair_idx2 = random.randint(0, 5)
            if pair_idx1 == pair_idx2:
                pair_idx2 = (pair_idx2 + 1) % 6
            v1, v2 = verts[pair_idx1], verts[pair_idx2]
            if v1 > v2:
                v1, v2 = v2, v1
            
            if not hasattr(construct, 'edge_lookup_29'):
                lookup = {}
                idx = 0
                for a in range(29):
                    for b in range(a + 1, 29):
                        lookup[(a, b)] = idx
                        idx += 1
                construct.edge_lookup_29 = lookup
            
            edge_idx = construct.edge_lookup_29[(v1, v2)]
            colors[edge_idx] = 1 - colors[edge_idx]

        return {'n': 29, 'colors': colors}

# Ensure the function is defined
if __name__ == "__main__":
    result = construct()
    print(result['n'])
