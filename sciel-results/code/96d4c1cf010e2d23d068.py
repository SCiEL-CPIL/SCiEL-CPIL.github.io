"""
4-coloring of K_n with no monochromatic triangle (R(3,3,3,3)).

Strategy:
  1. Build the known K_32 seed (two Clebsch copies, inter-copy edges color 3).
  2. Use a vertex-signature formulation: assign each vertex a 4-bit signature,
     edge color = f(sig_i XOR sig_j). This makes the search space 4^n instead of
     4^(n(n-1)/2), enabling much faster local search.
  3. Run multi-start local search on vertex signatures to minimize monochromatic
     triangles, with bitmask-accelerated triangle counting.
  4. If zero triangles are found for n > 32, return that coloring.
"""

import time
import random
import numpy as np


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin before 150 s kill

    random.seed(42)
    np.random.seed(42)

    # ---------- GF(16) helpers for the Clebsch seed ----------
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def build_seed_32():
        """Return list of colors for K_32 in lexicographic edge order."""
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(
                    log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                )
        return colors

    # ---------- Edge index helper ----------
    def edge_idx(i, j, n):
        """Lexicographic edge index for (i,j) in K_n."""
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    # ---------- Vertex-signature based triangle counting ----------
    def count_tris_from_sigs(sigs, n):
        """
        Count monochromatic triangles given vertex signatures.
        Edge color between vertices i,j = color_map[sigs[i] ^ sigs[j]]
        where color_map maps 4-bit XOR values to colors 0-3.
        """
        # Precompute color for each XOR value (0..15)
        # We use a fixed mapping: color = popcount(xor) % 4
        # Actually, let's use a parameterized mapping. For now use popcount%4.
        # But that might not work well. Let's use: color = (xor & 3) | ((xor >> 2) & 3) % 4
        # Simpler: just use a lookup table indexed by XOR value.
        # We'll use color = popcount(xor) % 4 for now, but this is fixed.
        
        # Actually, let's think differently. We want edge_color(i,j) to be in {0,1,2,3}.
        # A natural choice: partition the 16 possible XOR values into 4 groups of 4.
        # Group c contains XOR values where the "color function" gives c.
        
        # For the Clebsch graph connection: the Clebsch graph uses GF(16) and
        # color = log[xor] % 3 for within-copy edges. That's 3 colors.
        # For 4 colors, we can use: color = log[xor] % 4 if xor != 0, else 0.
        # But log[0] is undefined. Let's handle xor=0 separately (same signature -> color 0).
        
        # Actually, let me use a simpler approach:
        # color(xor) = popcount(xor) % 4
        # This gives a valid 4-coloring of the "XOR graph" but may have triangles.
        
        # Let's just count triangles directly.
        color_of_xor = [0] * 16
        for v in range(16):
            color_of_xor[v] = bin(v).count('1') % 4  # popcount mod 4
        
        # Count triangles: for each triple (i,j,k) with i<j<k, check if all three
        # edges have the same color.
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = color_of_xor[sigs[i] ^ sigs[j]]
                for k in range(j + 1, n):
                    if color_of_xor[sigs[i] ^ sigs[k]] == c1 and \
                       color_of_xor[sigs[j] ^ sigs[k]] == c1:
                        total += 1
        return total

    # ---------- Faster triangle count using adjacency bitmasks ----------
    def build_adj_from_sigs(sigs, n):
        """Build adjacency bitmasks from vertex signatures."""
        color_of_xor = [bin(v).count('1') % 4 for v in range(16)]
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = color_of_xor[sigs[i] ^ sigs[j]]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def count_tris_adj(adj, n):
        """Count monochromatic triangles from adjacency bitmasks."""
        total = 0
        for c in range(4):
            for i in range(n):
                ai = adj[c][i]
                mask = ai & ~((1 << (i + 1)) - 1)
                while mask:
                    lb = mask & (-mask)
                    j = lb.bit_length() - 1
                    total += (adj[c][i] & adj[c][j]).bit_count()
                    mask ^= lb
        return total // 3

    # ---------- Local search on vertex signatures ----------
    def local_search_sigs(sigs, n, time_limit, max_iter_per_vertex=50):
        """
        Local search: try changing one vertex's signature at a time.
        Returns (best_sigs, best_tri).
        """
        color_of_xor = [bin(v).count('1') % 4 for v in range(16)]
        adj = build_adj_from_sigs(sigs, n)
        tri = count_tris_adj(adj, n)
        best_tri = tri
        best_sigs = sigs[:]
        
        while time.time() < time_limit:
            if tri == 0:
                break
            
            # Pick a random vertex to reassign
            v = random.randint(0, n - 1)
            old_sig = sigs[v]
            new_sig = random.randint(0, 15)
            if new_sig == old_sig:
                continue
            
            # Compute delta: remove old contributions, add new
            # For vertex v, its edges to all other vertices change color.
            # Triangles involving v: for each pair (a,b) where a,b != v,
            # check if edges (v,a), (v,b), (a,b) were monochromatic.
            
            # This is expensive to do incrementally. Let's just recompute
            # triangles involving v.
            
            # Actually, let's do a simpler approach: recompute total triangles
            # but only count those involving v, then adjust.
            
            # Triangles involving v with old sig:
            old_tri_v = 0
            for a in range(n):
                if a == v:
                    continue
                ca = color_of_xor[old_sig ^ sigs[a]]
                for b in range(a + 1, n):
                    if b == v:
                        continue
                    cb = color_of_xor[old_sig ^ sigs[b]]
                    if ca == cb:
                        cab = color_of_xor[sigs[a] ^ sigs[b]]
                        if cab == ca:
                            old_tri_v += 1
            old_tri_v //= 1  # each triangle counted once per vertex in it? No.
            # Actually each triangle with v is counted once when we iterate a,b.
            # But we need to be careful. Let me just recompute fully for simplicity
            # when n is small enough.
            
            # For n <= 50, full recompute is O(n^3) = 125000 ops, which is fast.
            # Let's just do full recompute.
            
            sigs[v] = new_sig
            adj_new = build_adj_from_sigs(sigs, n)
            tri_new = count_tris_adj(adj_new, n)
            
            if tri_new <= tri:
                tri = tri_new
                adj = adj_new
                if tri < best_tri:
                    best_tri = tri
                    best_sigs = sigs[:]
            else:
                # Reject (or accept with small probability for exploration)
                if random.random() < 0.1:
                    # Keep the change for exploration
                    tri = tri_new
                    adj = adj_new
                else:
                    sigs[v] = old_sig
        
        return best_sigs, best_tri

    # ---------- Convert vertex signatures to edge colors ----------
    def sigs_to_colors(sigs, n):
        """Convert vertex signatures to edge color list in lexicographic order."""
        color_of_xor = [bin(v).count('1') % 4 for v in range(16)]
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(color_of_xor[sigs[i] ^ sigs[j]])
        return colors

    # ---------- Step 1: Build seed K_32 ----------
    seed_colors = build_seed_32()
    best_n = 32
    best_result = {'n': 32, 'colors': seed_colors}

    # ---------- Step 2: Try vertex-signature local search for larger n ----------
    # For each target n, try multiple random initializations.
    
    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 36, 34]
    
    for target_n in targets:
        if target_n <= best_n:
            continue
        if time.time() > DEADLINE - 15:
            break
        
        # Try multiple restarts
        for restart in range(8):
            if time.time() > DEADLINE - 10:
                break
            
            # Random initialization of vertex signatures
            sigs = [random.randint(0, 15) for _ in range(target_n)]
            
            # Run local search with a time slice
            time_slice = min(10.0, (DEADLINE - 10 - time.time()) / max(1, 8 - restart))
            best_sigs, best_tri = local_search_sigs(sigs, target_n, time_limit=time.time() + time_slice)
            
            if best_tri == 0:
                colors = sigs_to_colors(best_sigs, target_n)
                return {'n': target_n, 'colors': colors}
            
            # Also try a "structured" initialization based on the Clebsch pattern
            # for the first few restarts
            if restart < 2:
                # Initialize first 32 vertices with Clebsch-like structure
                clebsch_sigs = []
                for i in range(32):
                    if i < 16:
                        clebsch_sigs.append(i)  # identity in GF(16)
                    else:
                        clebsch_sigs.append(i - 16 + 16)  # second copy
                # Actually this doesn't map well. Let's just use random for all.
                pass

    # ---------- Step 3: Edge-based SA as fallback (from original) ----------
    def greedy_init(n):
        """Greedy initialization that tries to avoid triangles."""
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * (n * (n - 1) // 2)
        
        for i in range(n):
            for j in range(i + 1, n):
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    common = adj[k][i] & adj[k][j]
                    delta = common.bit_count()
                    if delta < best_delta or (delta == best_delta and random.random() < 0.5):
                        best_delta = delta
                        best_k = k
                
                idx = edge_idx(i, j, n)
                colors[idx] = best_k
                adj[best_k][i] |= (1 << j)
                adj[best_k][j] |= (1 << i)
        
        tri = count_tris_adj(adj, n)
        return adj, colors, tri

    def sa_color(adj, colors, n, time_limit):
        """
        Run simulated annealing to minimize monochromatic triangles.
        Returns (colors_list, n_triangles).
        """
        tri = count_tris_adj(adj, n)
        if tri == 0:
            return colors[:], 0

        T = 1.5
        T_min = 0.05
        cooling_rate = 0.9997
        best_tri = tri
        best_colors = colors[:]

        n_edges = n * (n - 1) // 2
        edge_to_ij = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_ij[idx] = (i, j)
                idx += 1

        while time.time() < time_limit:
            if tri == 0:
                break

            ei = random.randint(0, n_edges - 1)
            i, j = edge_to_ij[ei]

            c_old = colors[ei]
            c_new = (c_old + 1 + random.randint(0, 2)) % 4

            k_old = (adj[c_old][i] & adj[c_old][j]).bit_count()
            k_new = (adj[c_new][i] & adj[c_new][j]).bit_count()
            delta = k_new - k_old

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                if T > T_min:
                    p = np.exp(-delta / T)
                    if random.random() < p:
                        accept = True

            if accept:
                colors[ei] = c_new
                adj[c_old][i] &= ~(1 << j)
                adj[c_old][j] &= ~(1 << i)
                adj[c_new][i] |= (1 << j)
                adj[c_new][j] |= (1 << i)
                tri += delta
                if tri < best_tri:
                    best_tri = tri
                    best_colors = colors[:]
                    if best_tri == 0:
                        break

            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_colors, best_tri

    # Try edge-based SA for targets
    for target_n in [50, 49, 48, 47, 46]:
        if target_n <= best_n:
            continue
        if time.time() > DEADLINE - 10:
            break
        
        for restart in range(3):
            if time.time() > DEADLINE - 5:
                break
            
            if restart % 2 == 0:
                adj, init, tri = greedy_init(target_n)
            else:
                # Random init
                adj = [[0] * target_n for _ in range(4)]
                init = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        k = random.randint(0, 3)
                        idx = edge_idx(i, j, target_n)
                        init.append(k)
                        adj[k][i] |= (1 << j)
                        adj[k][j] |= (1 << i)
            
            colors, tri = sa_color(adj, init, target_n, time_limit=DEADLINE - 5)
            if colors is not None and tri == 0:
                return {'n': target_n, 'colors': colors}

    return best_result
