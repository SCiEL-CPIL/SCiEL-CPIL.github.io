"""Rewritten solver: precomputed constraints + greedy + optimized local search."""
import numpy as np
import time

def construct():
    t0 = time.time()
    N = 128
    
    # Precompute: for each position p (0-based), all APs it participates in, with roles.
    # ap_involving[p] = list of (role, other1, other2, other3) where role is 0,1,2,3
    # and others are the other 3 positions (0-based)
    
    ap_involving = [[] for _ in range(N)]
    
    # Generate all 4-APs: (a, a+d, a+2d, a+3d), 1-based, d>=1, a+3d<=128
    all_aps = []
    for d in range(1, 43):  # a+3d <= 128, a>=1, so 3d <= 127, d <= 42
        for a in range(1, 129 - 3*d):
            positions = [a-1, a+d-1, a+2*d-1, a+3*d-1]
            all_aps.append(positions)
            for role, pos in enumerate(positions):
                others = [positions[i] for i in range(4) if i != role]
                ap_involving[pos].append((role, others[0], others[1], others[2]))
    
    # Greedy: place positions in order, choose color minimizing violations in APs where this is the last (max) position
    # For position p, APs where p is the max: those where p = a+3d
    last_aps = [[] for _ in range(N)]
    for p in range(N):
        pos = p + 1
        for d in range(1, pos//3 + 1):
            a = pos - 3*d
            if a < 1:
                break
            o1 = a - 1
            o2 = a + d - 1
            o3 = a + 2*d - 1
            last_aps[p].append((o1, o2, o3))
    
    def greedy(n_colors):
        colors = np.full(N, -1, dtype=np.int8)
        for p in range(N):
            best_color = 0
            best_viol = None
            for c in range(n_colors):
                viol = 0
                for (o1, o2, o3) in last_aps[p]:
                    c1, c2, c3 = int(colors[o1]), int(colors[o2]), int(colors[o3])
                    distinct = len({c, c1, c2, c3})
                    if distinct < 3:
                        viol += 1
                if best_viol is None or viol < best_viol:
                    best_viol = viol
                    best_color = c
            colors[p] = best_color
        return colors
    
    def count_violations(colors):
        total = 0
        for ap in all_aps:
            distinct = len({int(colors[ap[0]]), int(colors[ap[1]]), int(colors[ap[2]]), int(colors[ap[3]])})
            if distinct < 3:
                total += 1
        return total
    
    # Start with 3 colors
    colors = greedy(3)
    best_viol = count_violations(colors)
    best_colors = colors.copy()
    
    deadline = t0 + 75
    
    # Local search: for each position, try all colors, keep if improves
    # Optimize: only recompute violations for APs involving the changed position
    def compute_violations_for_position(p, colors):
        """Compute number of violations in APs involving position p."""
        viol = 0
        for (role, o1, o2, o3) in ap_involving[p]:
            cp = int(colors[p])
            co1, co2, co3 = int(colors[o1]), int(colors[o2]), int(colors[o3])
            distinct = len({cp, co1, co2, co3})
            if distinct < 3:
                viol += 1
        return viol
    
    # Local search
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        for p in range(N):
            if time.time() > deadline:
                break
            current_color = int(best_colors[p])
            for c in range(3):
                if c == current_color:
                    continue
                best_colors[p] = c
                # Compute change in violations
                # Old violations for APs involving p
                best_colors[p] = current_color
                old_viol = compute_violations_for_position(p, best_colors)
                best_colors[p] = c
                new_viol = compute_violations_for_position(p, best_colors)
                if new_viol < old_viol:
                    best_viol += (new_viol - old_viol)
                    improved = True
                    break
                else:
                    best_colors[p] = current_color
            if best_viol == 0:
                break
        if best_viol == 0:
            break
    
    # If still violations, try 4 colors
    if best_viol > 0 and time.time() < deadline:
        colors4 = greedy(4)
        viol4 = count_violations(colors4)
        if viol4 < best_viol:
            best_colors = colors4
            best_viol = viol4
            # Local search with 4 colors
            for iteration in range(300):
                if time.time() > deadline:
                    break
                improved = False
                for p in range(N):
                    if time.time() > deadline:
                        break
                    current_color = int(best_colors[p])
                    for c in range(4):
                        if c == current_color:
                            continue
                        best_colors[p] = c
                        best_colors[p] = current_color
                        old_viol = compute_violations_for_position(p, best_colors)
                        best_colors[p] = c
                        new_viol = compute_violations_for_position(p, best_colors)
                        if new_viol < old_viol:
                            best_viol += (new_viol - old_viol)
                            improved = True
                            break
                        else:
                            best_colors[p] = current_color
                    if best_viol == 0:
                        break
                if best_viol == 0:
                    break
    
    result = [int(c) for c in best_colors]
    return result
