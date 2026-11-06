import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        """Compute edge index for (i,j) in lexicographic order for K_n."""
        if i > j:
            i, j = j, i
        # Edges are: (0,1),(0,2),...,(0,n-1),(1,2),(1,3),...,(1,n-1),...
        # Index = sum_{k=0}^{i-1} (n-1-k) + (j-1-i)
        # = i*(n-1) - i*(i-1)//2 + j - 1 - i
        # = i*(n-1) - i*(i-1)//2 + j - 1 - i
        # Simplify: i*(n-1) - i*(i-1)//2 - i + j - 1
        # = i*(n-2) - i*(i-1)//2 + j - 1
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles(n):
        """Build list of all C6 cycles as lists of edge indices."""
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                # Enforce canonical form: perm[0] < perm[-1] to avoid reverse duplicates
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles.append(cycle_edges)
        return cycles

    def count_c6_for_color(colors, n, color, cycles):
        """Count monochromatic C6s for a specific color."""
        count = 0
        for ce in cycles:
            if all(colors[e] == color for e in ce):
                count += 1
        return count

    def total_c6(colors, n, cycles):
        """Count all monochromatic C6s."""
        total = 0
        for ce in cycles:
            c0 = colors[ce[0]]
            if all(colors[e] == c0 for e in ce):
                total += 1
        return total

    def spectral_init(n, seed_offset=0):
        """Generate initial coloring using spectral graph theory approach."""
        rng = np.random.default_rng(42 + seed_offset)
        
        # Create a random regular graph adjacency matrix
        deg = min(4, n - 1)  # 4-regular if possible
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            # Pick deg random neighbors
            candidates = [j for j in range(n) if j != i]
            neighbors = rng.choice(candidates, size=deg, replace=False)
            for j in neighbors:
                A[i, j] = 1.0
        
        # Compute Laplacian
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Get top 4 eigenvectors (excluding trivial ones)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Sort by eigenvalue (ascending), skip the first (trivial)
            idx = np.argsort(eigvals)[1:5]
            V = eigvecs[:, idx]
        except Exception:
            # Fallback to random
            return [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Assign colors based on quadrant signs of the 4 eigenvector components
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # Get the 4-dimensional vectors for vertices i and j
                vi = V[i, :]
                vj = V[j, :]
                # Compute dot product and cross-like features to determine quadrant
                dot = np.dot(vi, vj)
                # Use sign patterns of differences
                diff = vi - vj
                signs = np.sign(diff)
                # Map to color 0-3 based on dominant sign pattern
                # Simple heuristic: use first two dimensions
                s1 = 1 if signs[0] >= 0 else 0
                s2 = 1 if signs[1] >= 0 else 0
                color = s1 * 2 + s2
                colors.append(int(color))
        
        return colors

    def search_n(n, time_budget, seed_offset=0, use_spectral=False):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        # Build cycles
        cycles = build_cycles(n)
        num_cycles = len(cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        # Build edge-to-cycles mapping
        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)

        # Initial coloring: use spectral if requested, else random
        if use_spectral:
            colors = spectral_init(n, seed_offset)
        else:
            colors = [random.randint(0, 3) for _ in range(m)]
        bad_count = total_c6(colors, n, cycles)

        if bad_count == 0:
            return {'n': n, 'colors': colors}

        best_colors = colors[:]
        best_bad = bad_count

        # Simulated annealing with better parameters
        T_start = 1.5
        T_end = 0.01
        max_iter = 200000
        start_iter_time = time.time()

        # Precompute for faster delta calculation
        # For each edge, store the cycles it's in
        # Use a more efficient delta calculation

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]

            # Pick new color (uniform from other 3)
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)

            # Calculate delta efficiently
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                # Count how many edges in this cycle have color c_old (excluding e)
                # and how many have color c_new (excluding e)
                # A cycle is monochromatic with color c if all 6 edges have color c
                # After changing e from c_old to c_new:
                # - If before: all 6 were c_old => it was mono (contributes +1 to bad)
                #   After: e is c_new, so it's no longer mono with c_old, 
                #   but could be mono with c_new if other 5 are also c_new
                # - We need to track per-color

                # Faster: for this cycle, count colors of the other 5 edges
                cnt_old = 0
                cnt_new = 0
                for ed in ce:
                    if ed == e:
                        continue
                    if colors[ed] == c_old:
                        cnt_old += 1
                    elif colors[ed] == c_new:
                        cnt_new += 1

                # Before change:
                # Cycle was mono with c_old if all 6 edges are c_old, i.e., cnt_old == 5 (since e was c_old)
                # Cycle was mono with c_new if all 6 edges are c_new, i.e., cnt_new == 5 (impossible since e was c_old)
                # So before: mono_count_old = 1 if cnt_old == 5 else 0; mono_count_new = 0
                # After change:
                # Cycle is mono with c_old if cnt_old == 5 and c_new == c_old (impossible since c_new != c_old)
                #   Actually after, e has color c_new, so for mono with c_old, need all 6 = c_old, 
                #   but e is c_new != c_old, so never mono with c_old after.
                #   Wait: before, e was c_old. After, e is c_new.
                #   Mono with c_old after: need all edges = c_old. e is c_new != c_old, so NO.
                #   Mono with c_new after: need all edges = c_new. e is c_new. Other 5 need to be c_new.
                #   So mono with c_new after: 1 if cnt_new == 5 else 0.
                #   Mono with c_old before: 1 if cnt_old == 5 else 0.
                #   Mono with c_new before: e was c_old, so for all 6 to be c_new, need c_old == c_new (impossible). So 0.

                before_mono = 1 if cnt_old == 5 else 0
                after_mono = 1 if cnt_new == 5 else 0

                delta += (after_mono - before_mono)

            # Acceptance
            if delta <= 0:
                accept = True
            else:
                # Temperature schedule
                progress = i / max_iter
                T = T_start * (T_end / T_start) ** progress
                if delta / T > 50:
                    accept = False
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors}

        # Return best found if it's valid
        if best_bad == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]

    # Allocate time: prioritize higher n
    # n=17: 25s, n=16: 20s, n=15: 15s, n=14: 12s, n=13: 10s, n=12: 8s, n=11: 7s, n=10: 5s, rest: 3s each
    time_alloc = {17: 25, 16: 20, 15: 15, 14: 12, 13: 10, 12: 8, 11: 7, 10: 5, 9: 3, 8: 3, 7: 2, 6: 2}

    for n in targets:
        if time.time() > deadline - 8:
            break

        tb = time_alloc.get(n, 5)
        # Try more restarts for harder n, alternating between spectral and random init
        num_restarts = 3 if n >= 15 else 2
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 8:
                break
            # Alternate between spectral and random initialization
            use_spectral = (r % 2 == 0)
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n, use_spectral=use_spectral)
            if result:
                best_result = result
                # Found valid coloring for n, return immediately since we go high to low
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
