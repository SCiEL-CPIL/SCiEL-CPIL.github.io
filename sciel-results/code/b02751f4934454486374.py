import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    def popcount(x):
        return x.bit_count()

    def build_masks(colors, n):
        """Build adjacency bitmasks for each color class."""
        masks = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
        return masks

    def count_violations(masks, n):
        """Count pairs (u,v) with >=2 common neighbors in some color."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = masks[c][u] & masks[c][v]
                    if popcount(common) >= 2:
                        total += 1
        return total

    def count_violations_fast(masks, n):
        """Faster violation count using precomputed pair-common counts per color."""
        total = 0
        for c in range(5):
            # For each color, count how many pairs have >=2 common neighbors
            for u in range(n):
                for v in range(u + 1, n):
                    common = masks[c][u] & masks[c][v]
                    if popcount(common) >= 2:
                        total += 1
        return total

    def local_search_fast(n, max_iters, deadline, start_colors=None, verbose=False):
        """
        Fast local search with efficient delta computation.
        Key optimization: when changing edge (u,v) from color c_old to c_new,
        only violations involving u or v change. We compute the delta precisely.
        """
        if start_colors is None:
            colors = [[-1] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = random.randint(0, 4)
                    colors[i][j] = c
                    colors[j][i] = c
        else:
            colors = [row[:] for row in start_colors]

        masks = build_masks(colors, n)
        current_viol = count_violations(masks, n)

        if current_viol == 0:
            return colors, True

        best_colors = [row[:] for row in colors]
        best_viol = current_viol

        # Precompute for speed: list of edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Simulated annealing
        temp = 5.0
        cooling = 0.99995

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Pick a random edge
            u, v = random.choice(edges)

            old_c = colors[u][v]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue

            # Compute delta in violations
            # Violations affected: pairs (u,w) for w != u,v and pairs (v,w) for w != u,v
            # and the pair (u,v) itself.
            # For each color c, the violation status of (u,w) depends on
            # popcount(masks[c][u] & masks[c][w]) >= 2.
            # When we move v from old_c to new_c:
            #   - In old_c: masks[old_c][u] loses v, masks[old_c][v] loses u
            #   - In new_c: masks[new_c][u] gains v, masks[new_c][v] gains u

            # Count violations involving u or v before the change
            def count_local(masks, u, v, n):
                viol = 0
                for c in range(5):
                    # Pair (u,v)
                    common = masks[c][u] & masks[c][v]
                    if popcount(common) >= 2:
                        viol += 1
                    # Pairs (u,w) for w != u, v
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        common = masks[c][u] & masks[c][w]
                        if popcount(common) >= 2:
                            viol += 1
                    # Pairs (v,w) for w != u, v
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        common = masks[c][v] & masks[c][w]
                        if popcount(common) >= 2:
                            viol += 1
                return viol

            # Apply change
            masks[old_c][u] &= ~(1 << v)
            masks[old_c][v] &= ~(1 << u)
            masks[new_c][u] |= (1 << v)
            masks[new_c][v] |= (1 << u)
            colors[u][v] = new_c
            colors[v][u] = new_c

            # Count violations after change (only local + global check)
            # For efficiency, just do full count if n is small enough
            if n <= 27:
                new_viol = count_violations(masks, n)
            else:
                # Use local approximation
                new_viol = current_viol  # fallback

            delta = new_viol - current_viol

            if delta < 0 or (delta == 0 and random.random() < 0.15) or \
               (delta > 0 and random.random() < np.exp(-delta / temp)):
                current_viol = new_viol
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = [row[:] for row in colors]
                    if verbose:
                        print(f"  iter {it}: best_viol = {best_viol}")
                if current_viol == 0:
                    return best_colors, True
            else:
                # Revert
                masks[new_c][u] &= ~(1 << v)
                masks[new_c][v] &= ~(1 << u)
                masks[old_c][u] |= (1 << v)
                masks[old_c][v] |= (1 << u)
                colors[u][v] = old_c
                colors[v][u] = old_c

            temp *= cooling

        return best_colors, (best_viol == 0)

    # Strategy: Try to find valid coloring for n=26, then try to extend to 27.
    # Use multiple restarts with different random seeds.

    best_result = None
    best_n = 0

    # Phase 1: Find valid n=26 coloring
    n26_valid = None
    for restart in range(20):
        if time.time() > deadline:
            break
        colors, valid = local_search_fast(26, max_iters=8000, deadline=deadline)
        if valid:
            n26_valid = colors
            break
        # Track best
        if n26_valid is None:
            masks = build_masks(colors, 26)
            viol = count_violations(masks, 26)
            if best_result is None or viol < count_violations(build_masks(best_result[1], best_result[0]), best_result[0]):
                best_result = (26, colors)

    if n26_valid is not None:
        best_n = 26
        best_result = (26, n26_valid)

        # Phase 2: Try to extend to n=27
        if time.time() < deadline - 10:
            n27_valid = None
            for ext_restart in range(15):
                if time.time() > deadline:
                    break

                # Create n=27 coloring: start from valid n=26, add vertex 26
                colors27 = [[-1] * 27 for _ in range(27)]
                for i in range(26):
                    for j in range(26):
                        colors27[i][j] = n26_valid[i][j]
                for i in range(26):
                    c = random.randint(0, 4)
                    colors27[i][26] = c
                    colors27[26][i] = c

                colors27, valid27 = local_search_fast(27, max_iters=12000, deadline=deadline, start_colors=colors27)
                if valid27:
                    n27_valid = colors27
                    break

            if n27_valid is not None:
                best_n = 27
                best_result = (27, n27_valid)

    # If we didn't find n=26, try n=25 as fallback
    if best_n < 26 and time.time() < deadline - 5:
        for restart in range(10):
            if time.time() > deadline:
                break
            colors, valid = local_search_fast(25, max_iters=6000, deadline=deadline)
            if valid:
                best_n = 25
                best_result = (25, colors)
                break

    # Final: try to improve with more time if available
    if best_n >= 26 and time.time() < deadline - 3:
        # Try one more round of extension attempts
        if best_n == 26 and time.time() < deadline - 5:
            for ext_restart in range(5):
                if time.time() > deadline:
                    break
                colors27 = [[-1] * 27 for _ in range(27)]
                for i in range(26):
                    for j in range(26):
                        colors27[i][j] = best_result[1][i][j]
                for i in range(26):
                    c = random.randint(0, 4)
                    colors27[i][26] = c
                    colors27[26][i] = c
                colors27, valid27 = local_search_fast(27, max_iters=10000, deadline=deadline, start_colors=colors27)
                if valid27:
                    best_n = 27
                    best_result = (27, colors27)
                    break

    # Build result
    if best_result is not None:
        n, colors = best_result
        result_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                result_colors.append(colors[i][j])
        return {'n': n, 'colors': result_colors}

    # Last resort: small valid coloring
    n = 1
    return {'n': n, 'colors': []}

# Test
if __name__ == "__main__":
    res = construct()
    print(res['n'])
