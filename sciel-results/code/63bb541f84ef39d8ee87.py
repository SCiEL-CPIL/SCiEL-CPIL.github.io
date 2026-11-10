# EVOLVE-BLOCK-START
import time
import random

# Hex-lattice neighbors (unit-distance moves)
NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))

def _edges_of(sel, nbr):
    """Count unit-distance pairs among the selected points."""
    c = 0
    for i in sel:
        for nb in nbr[i]:
            if nb > i and nb in sel:
                c += 1
    return c

def construct():
    start = time.time()
    DL = 140.0            # hard safety margin before the 150 s kill
    def dl():
        return time.time() < DL

    # ---- Build the search space: all lattice points inside a hexagon of radius R ----
    R = 6
    cand = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if max(abs(a), abs(b), abs(a + b)) <= R:
                cand.append((a, b))
    N = len(cand)
    idx = {p: i for i, p in enumerate(cand)}
    nbr = [set() for _ in range(N)]
    for i, (a, b) in enumerate(cand):
        for da, db in NEIGH:
            j = idx.get((a + da, b + db))
            if j is not None and j != i:
                nbr[i].add(j)
                nbr[j].add(i)

    NSEL = 32
    
    # ---- Deterministic, objective-guided construction of an initial 32-set ----
    # Start with a compact core
    core = set(i for i in range(N)
               if max(abs(cand[i][0]), abs(cand[i][1]), abs(cand[i][0] + cand[i][1])) <= 2)
    sel = set(core)
    # Greedily add points with highest degree to current sel
    while len(sel) < NSEL:
        best = None
        bg = -1
        for i in range(N):
            if i in sel:
                continue
            g = len(nbr[i] & sel)
            if g > bg:
                bg = g
                best = i
        if best is None:
            break
        sel.add(best)
    if len(sel) < NSEL:
        for i in range(N):
            if i not in sel:
                sel.add(i)
            if len(sel) == NSEL:
                break

    best_edges = _edges_of(sel, nbr)
    best_sel = set(sel)
    
    random.seed(42)
    
    # Local search with 1-swap and fast 2-swap
    for restart in range(20):
        if not dl():
            break
            
        if restart > 0:
            sel = set(best_sel)
            # Perturb
            for _ in range(5):
                x = random.choice(list(sel))
                y = random.randint(0, N - 1)
                while y in sel:
                    y = random.randint(0, N - 1)
                sel.remove(x)
                sel.add(y)
        
        cur = _edges_of(sel, nbr)
        if cur > best_edges:
            best_edges = cur
            best_sel = set(sel)
        
        improved = True
        while improved and dl():
            improved = False
            
            # Precompute degrees in sel for fast delta calculation
            deg_in_sel = [0] * N
            for i in sel:
                deg_in_sel[i] = len(nbr[i] & sel)
            
            # 1-Swap
            for x in list(sel):
                if not dl(): break
                gx = deg_in_sel[x]
                for y in range(N):
                    if y in sel: continue
                    gy = deg_in_sel[y]
                    # Delta = (gy - edges(y, sel\{x})) + (gx - edges(x, sel\{y})) - 2*overlap?
                    # Simpler:
                    # New edges = (old edges - edges incident to x or y) + edges incident to y in new set
                    # Let's use the property:
                    # delta = (deg_new(y) + deg_new(x_removed)) ...
                    # Actually, for 1-swap:
                    # Remove x, add y.
                    # Lost: edges from x to sel\{x} = gx
                    # Gained: edges from y to sel\{x} = gy - (1 if x in nbr[y] else 0)
                    # Wait, x is removed, so y connects to sel\{x}.
                    # gy counts edges from y to sel. If x in nbr[y], one of those is to x.
                    # So gained = gy - (1 if x in nbr[y] else 0)
                    # But we also lose edges from x to sel\{x}.
                    # So delta = (gy - (1 if x in nbr[y] else 0)) - gx
                    if x in nbr[y]:
                        d = gy - 1 - gx
                    else:
                        d = gy - gx
                    if d > 0:
                        sel.remove(x)
                        sel.add(y)
                        cur += d
                        improved = True
                        if cur > best_edges:
                            best_edges = cur
                            best_sel = set(sel)
                        if best_edges >= 76:
                            return [[cand[i][0], cand[i][1]] for i in sorted(best_sel)]
                if best_edges >= 76:
                    break
            if best_edges >= 76:
                break

            # 2-Swap (fast)
            if not dl(): continue
            sel_list = list(sel)
            not_sel_list = [i for i in range(N) if i not in sel]
            
            # Precompute degrees again if changed by 1-swap? 
            # If 1-swap improved, sel changed. We should recompute deg_in_sel if improved.
            if improved:
                deg_in_sel = [0] * N
                for i in sel:
                    deg_in_sel[i] = len(nbr[i] & sel)
            
            for i1 in range(len(sel_list)):
                if not dl(): break
                x1 = sel_list[i1]
                gx1 = deg_in_sel[x1]
                for i2 in range(i1 + 1, len(sel_list)):
                    x2 = sel_list[i2]
                    gx2 = deg_in_sel[x2]
                    edge_x1x2 = 1 if x2 in nbr[x1] else 0
                    lost = gx1 + gx2 - edge_x1x2
                    
                    for y1 in not_sel_list:
                        gy1 = deg_in_sel[y1]
                        for y2 in not_sel_list:
                            if y2 == y1: continue
                            gy2 = deg_in_sel[y2]
                            edge_y1y2 = 1 if y2 in nbr[y1] else 0
                            
                            # New degrees for y1, y2 in sel' = sel - {x1, x2} + {y1, y2}
                            # deg_new(y1) = gy1 - (1 if x1 in nbr[y1] else 0) - (1 if x2 in nbr[y1] else 0) + (1 if y2 in nbr[y1] else 0)
                            # deg_new(y2) = gy2 - (1 if x1 in nbr[y2] else 0) - (1 if x2 in nbr[y2] else 0) + (1 if y1 in nbr[y2] else 0)
                            # Gained = deg_new(y1) + deg_new(y2) - edge_y1y2 (because edge y1-y2 is counted twice in deg sums? No, deg_new counts edges to neighbors in sel'.
                            # Edge y1-y2 is in sel'. It is counted in deg_new(y1) and deg_new(y2).
                            # So sum of degrees counts it twice. We want to count it once.
                            # So Gained = deg_new(y1) + deg_new(y2) - edge_y1y2
                            
                            y1x1 = 1 if x1 in nbr[y1] else 0
                            y1x2 = 1 if x2 in nbr[y1] else 0
                            y2x1 = 1 if x1 in nbr[y2] else 0
                            y2x2 = 1 if x2 in nbr[y2] else 0
                            
                            deg_new_y1 = gy1 - y1x1 - y1x2 + edge_y1y2
                            deg_new_y2 = gy2 - y2x1 - y2x2 + edge_y1y2
                            
                            gained = deg_new_y1 + deg_new_y2 - edge_y1y2
                            
                            d2 = gained - lost
                            if d2 > 0:
                                sel.discard(x1)
                                sel.discard(x2)
                                sel.add(y1)
                                sel.add(y2)
                                cur += d2
                                improved = True
                                if cur > best_edges:
                                    best_edges = cur
                                    best_sel = set(sel)
                                if best_edges >= 76:
                                    return [[cand[i][0], cand[i][1]] for i in sorted(best_sel)]
                                break
                        if best_edges >= 76: break
                    if best_edges >= 76: break
                if best_edges >= 76: break
            if best_edges >= 76: break

    # ---- Finalize & validate ----
    sel = best_sel
    pts = [cand[i] for i in sorted(sel)]
    assert len(pts) == NSEL
    assert len(set(pts)) == NSEL
    for a, b in pts:
        assert abs(a) <= 64 and abs(b) <= 64
    return [[a, b] for a, b in pts]
# EVOLVE-BLOCK-END
