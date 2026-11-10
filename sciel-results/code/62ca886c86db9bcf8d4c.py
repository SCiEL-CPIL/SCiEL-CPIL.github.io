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

def _build_overlap(nbr, N):
    """Precompute overlap[i][j] = |N(i) ∩ N(j)| for all i, j."""
    ov = [[0] * N for _ in range(N)]
    for i in range(N):
        si = set(nbr[i])
        for j in range(N):
            if i == j:
                continue
            ov[i][j] = len(si & set(nbr[j]))
    return ov

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
    nbr = [[] for _ in range(N)]
    for i, (a, b) in enumerate(cand):
        for da, db in NEIGH:
            j = idx.get((a + da, b + db))
            if j is not None and j != i:
                nbr[i].append(j)

    NSEL = 32
    # ---- Precompute overlap matrix for fast 2-swap deltas ----
    ov = _build_overlap(nbr, N)
    # ---- Deterministic, objective-guided construction of an initial 32-set ----
    core = set(i for i in range(N)
               if max(abs(cand[i][0]), abs(cand[i][1]), abs(cand[i][0] + cand[i][1])) <= 2)
    sel = set(core)
    while len(sel) < NSEL:
        best = None
        bg = -1
        for i in range(N):
            if i in sel:
                continue
            g = sum(1 for nb in nbr[i] if nb in sel)
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
    cur = best_edges

    # ---- Fast single-swap local search with random restarts ----
    random.seed(42)
    
    for restart in range(10): # Limit restarts to ensure time safety
        if not dl():
            break
            
        # If not first restart, perturb the current best solution
        if restart > 0:
            sel = set(best_sel)
            # Randomly swap 5 points to escape local optimum
            sel_list = list(sel)
            for _ in range(5):
                x = random.choice(sel_list)
                # Find a random y not in sel
                while True:
                    y = random.randint(0, N - 1)
                    if y not in sel:
                        break
                sel.remove(x)
                sel.add(y)
        
        cur = _edges_of(sel, nbr)
        if cur > best_edges:
            best_edges = cur
            best_sel = set(sel)
        
        improved = True
        while improved and dl():
            improved = False
            # Single-swap moves
            for x in random.sample(list(sel), len(sel)):
                if not dl():
                    break
                gx = sum(1 for nb in nbr[x] if nb in sel)
                for y in range(N):
                    if y in sel:
                        continue
                    gy = sum(1 for nb in nbr[y] if nb in sel)
                    ovxy = ov[x][y]
                    d = gy + gx - 2 * ovxy
                    if d > 0:
                        sel.remove(x)
                        sel.add(y)
                        cur += d
                        improved = True
                        if cur > best_edges:
                            best_edges = cur
                            best_sel = set(sel)
                        if best_edges >= 76:
                            break
                if best_edges >= 76:
                    break
            if best_edges >= 76:
                break

            # Two-swap moves (x1,x2 out; y1,y2 in)
            if not dl():
                continue
            sel_list = list(sel)
            for i1 in range(len(sel_list)):
                if not dl():
                    break
                x1 = sel_list[i1]
                gx1 = sum(1 for nb in nbr[x1] if nb in sel)
                for i2 in range(i1 + 1, len(sel_list)):
                    x2 = sel_list[i2]
                    gx2 = sum(1 for nb in nbr[x2] if nb in sel)
                    # Edge between x1 and x2 (if any) is lost
                    edge_x1x2 = 1 if x2 in nbr[x1] else 0
                    for y1 in range(N):
                        if y1 in sel:
                            continue
                        gy1 = sum(1 for nb in nbr[y1] if nb in sel)
                        for y2 in range(N):
                            if y2 in sel or y2 == y1:
                                continue
                            gy2 = sum(1 for nb in nbr[y2] if nb in sel)
                            edge_y1y2 = 1 if y2 in nbr[y1] else 0
                            # Compute delta for removing x1,x2 and adding y1,y2
                            # Lost edges: gx1 + gx2 - edge_x1x2 (edges from x1,x2 to rest of sel)
                            # Gained edges: gy1 + gy2 - edge_y1y2 (edges from y1,y2 to rest of sel)
                            # But we must account for edges between {x1,x2} and {y1,y2}
                            # and between y1,y2 and the rest properly.
                            # Simpler: recompute delta as (new edges) - (old edges)
                            # Old edges involving x1 or x2: gx1 + gx2 - edge_x1x2
                            # New edges involving y1 or y2: gy1 + gy2 - edge_y1y2
                            # But gy1 counts edges from y1 to sel, which includes x1,x2
                            # So we need to subtract edges y1-x1, y1-x2, y2-x1, y2-x2 from gy1,gy2
                            # and add them back as part of the "new" configuration? No.
                            # Let's think: 
                            # Current edges = E
                            # After swap: E' = E - (edges incident to x1 or x2) + (edges incident to y1 or y2 in new set)
                            # edges incident to x1 or x2 in current set = gx1 + gx2 - edge_x1x2
                            # edges incident to y1 or y2 in new set (sel - {x1,x2} + {y1,y2}):
                            #   = (edges from y1 to sel - {x1,x2}) + (edges from y2 to sel - {x1,x2}) + edge_y1y2
                            #   = (gy1 - [x1 in nbr[y1]] - [x2 in nbr[y1]]) + (gy2 - [x1 in nbr[y2]] - [x2 in nbr[y2]]) + edge_y1y2
                            y1x1 = 1 if x1 in nbr[y1] else 0
                            y1x2 = 1 if x2 in nbr[y1] else 0
                            y2x1 = 1 if x1 in nbr[y2] else 0
                            y2x2 = 1 if x2 in nbr[y2] else 0
                            new_incident = (gy1 - y1x1 - y1x2) + (gy2 - y2x1 - y2x2) + edge_y1y2
                            old_incident = gx1 + gx2 - edge_x1x2
                            d2 = new_incident - old_incident
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
                                    break
                        if best_edges >= 76:
                            break
                    if best_edges >= 76:
                        break
                if best_edges >= 76:
                    break
            if best_edges >= 76:
                break

    # ---- Finalize & validate ----
    sel = best_sel
    pts = [cand[i] for i in sorted(sel)]
    assert len(pts) == NSEL
    assert len(set(pts)) == NSEL
    for a, b in pts:
        assert abs(a) <= 64 and abs(b) <= 64
    return [[a, b] for a, b in pts]
# EVOLVE-BLOCK-END
