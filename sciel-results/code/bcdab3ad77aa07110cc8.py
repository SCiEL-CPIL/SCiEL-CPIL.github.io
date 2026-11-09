"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import permutations

def _fits(x, y, z, w, h, d, placed, W, H, D):
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + w <= px or px + pw <= x):
            if not (y + h <= py or py + ph <= y):
                if not (z + d <= pz or pz + pd <= z):
                    return False
    return True

def load(boxes, container):
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = []

    for idx, (w, h, d) in enumerate(boxes):
        best = None
        seen = set()
        for perm in permutations((w, h, d)):
            if perm in seen:
                continue
            seen.add(perm)
            bw, bh, bd = perm
            # Generate candidate z-levels: 0 and top of each placed box
            z_candidates = [0]
            for (px, py, pz, pw, ph, pd) in placed:
                z_candidates.append(pz + pd)
            z_candidates = sorted(set(zz for zz in z_candidates if zz + bd <= D))
            
            for z in z_candidates:
                # Generate candidate x positions: 0 and right edge of each placed box
                x_candidates = [0]
                for (px, py, pz, pw, ph, pd) in placed:
                    x_candidates.append(px + pw)
                x_candidates = sorted(set(xx for xx in x_candidates if xx + bw <= W))
                
                for x in x_candidates:
                    # Generate candidate y positions: 0 and back edge of each placed box
                    y_candidates = [0]
                    for (px, py, pz, pw, ph, pd) in placed:
                        y_candidates.append(py + ph)
                    y_candidates = sorted(set(yy for yy in y_candidates if yy + bh <= H))
                    
                    for y in y_candidates:
                        if _fits(x, y, z, bw, bh, bd, placed, W, H, D):
                            key = (z, x, y)
                            if best is None or key < best[0]:
                                best = (key, bw, bh, bd)
                            break  # For this x, found best y (smallest y)
                    # Don't break x loop; try other x for same z to find better (z,x,y)
                # Don't break z loop; try other z for better overall
                
        if best is not None:
            (z, x, y), bw, bh, bd = best
            placed.append((x, y, z, bw, bh, bd))
            placements.append((0, x, y, z, bw, bh, bd))
        else:
            placements.append((-1, 0, 0, 0, w, h, d))

    return placements
# EVOLVE-BLOCK-END
