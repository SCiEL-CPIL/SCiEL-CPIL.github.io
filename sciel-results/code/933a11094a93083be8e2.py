import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    N = 128

    # Precompute all 4-term APs and 3-term APs for fast checking
    aps4 = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps4.append((i, i + d, i + 2 * d, i + 3 * d))

    aps3 = []
    for d in range(1, (N - 2) // 2 + 1):
        for i in range(0, N - 2 * d):
            aps3.append((i, i + d, i + 2 * d))

    # For each position, precompute which 4-term APs it participates in as the last element
    pos_aps4_last = [[] for _ in range(N)]
    pos_aps3_last = [[] for _ in range(N)]

    for (a, b, c, d_pos) in aps4:
        pos_aps4_last[d_pos].append((a, b, c))

    for (a, b, c) in aps3:
        pos_aps3_last[c].append((a, b))

    def is_valid_assignment(colors, pos, color):
        """Check if assigning 'color' to position 'pos' is valid given current colors."""
        for (a, b, c) in pos_aps4_last[pos]:
            if colors[a] == -1 or colors[b] == -1 or colors[c] == -1:
                continue
            distinct = len({color, colors[a], colors[b], colors[c]})
            if distinct < 3:
                return False
        for (a, b) in pos_aps3_last[pos]:
            if colors[a] == -1 or colors[b] == -1:
                continue
            if colors[a] == color and colors[b] == color:
                return False
        return True

    def forward_check(colors, num_colors):
        """Return (ok, domains) where domains[i] = set of valid colors for unassigned i."""
        domains = [None] * N
        for i in range(N):
            if colors[i] != -1:
                continue
            dom = set()
            for c in range(num_colors):
                if is_valid_assignment(colors, i, c):
                    dom.add(c)
            if not dom:
                return (False, None)
            domains[i] = dom
        return (True, domains)

    def mrv_pick(unassigned, domains):
        """Pick the unassigned position with fewest domain values (MRV)."""
        best = None
        best_size = None
        for i in unassigned:
            sz = len(domains[i])
            if best_size is None or sz < best_size:
                best_size = sz
                best = i
        return best

    def solve_csp(num_colors, max_restarts=50):
        """CSP solver with forward checking + MRV + backtracking."""
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                return None
            colors = [-1] * N
            for i in range(3):
                colors[i] = random.randint(0, num_colors - 1)

            ok, domains = forward_check(colors, num_colors)
            if not ok:
                continue

            unassigned = [i for i in range(N) if colors[i] == -1]

            def backtrack(colors, unassigned, domains):
                if time.time() - start_time > DEADLINE:
                    return False
                if not unassigned:
                    return True
                chosen = mrv_pick(unassigned, domains)
                for color in sorted(domains[chosen]):
                    colors[chosen] = color
                    new_unassigned = [i for i in unassigned if i != chosen]
                    ok, new_domains = forward_check(colors, num_colors)
                    if ok:
                        if backtrack(colors, new_unassigned, new_domains):
                            return True
                    colors[chosen] = -1
                return False

            if backtrack(colors, unassigned, domains):
                for (a, b, c, d_pos) in aps4:
                    distinct = len({colors[a], colors[b], colors[c], colors[d_pos]})
                    if distinct < 3:
                        return None
                return colors
        return None

    def polish(colors, num_colors, max_iters=500):
        """Reduce distinct colors by reassigning positions to consolidate colors."""
        current = list(colors)
        best = list(current)
        best_count = len(set(current))
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            if len(set(current)) >= best_count:
                break
            pos = random.randint(0, N - 1)
            counts = {}
            for c in current:
                counts[c] = counts.get(c, 0) + 1
            color_order = sorted(range(num_colors), key=lambda c: -counts.get(c, 0))
            improved = False
            for new_color in color_order:
                if new_color == current[pos]:
                    continue
                if not is_valid_assignment(current, pos, new_color):
                    continue
                old_color = current[pos]
                current[pos] = new_color
                if len(set(current)) < best_count:
                    best = list(current)
                    best_count = len(set(current))
                    improved = True
                    break
                else:
                    current[pos] = old_color
            if not improved:
                # No improvement this iteration; continue searching
                pass
        return best

    # Try with 3 colors first, then 4, etc.
    for num_colors in [3, 4, 5, 6, 7, 8]:
        if time.time() - start_time > DEADLINE:
            break
        result = solve_csp(num_colors)
        if result is not None:
            polished = polish(result, num_colors)
            return polished

    # Fallback: use the original greedy approach with more colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                   for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                       for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    return colors
