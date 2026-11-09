import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        perms = set(itertools.permutations([w, h, d]))
        return list(perms)

    box_orientations = [get_orientations(w, h, d) for (w, h, d) in boxes]

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    def greedy_pack(sort_key, pos_generator, rng=None, perturbation=0.0):
        """Greedy packing with given sort key and position generator."""
        indexed_boxes = sorted(range(n), key=lambda i: sort_key(boxes[i]), reverse=True)

        placed = []  # List of (idx, x, y, z, w, h, d)
        total_vol = 0

        for idx in indexed_boxes:
            if time.time() - start_time > DEADLINE:
                break

            best_placement = None
            best_score = float('inf')

            for ow, oh, od in box_orientations[idx]:
                candidate_positions = pos_generator(placed, ow, oh, od)

                for cx, cy, cz in candidate_positions:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if not valid:
                        continue

                    score = cx + cy + cz
                    if perturbation > 0 and rng is not None:
                        score += rng.random() * perturbation

                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)

            if best_placement:
                placed.append(best_placement)
                total_vol += best_placement[4] * best_placement[5] * best_placement[6]

        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        return result, total_vol

    def corners_generator(placed, ow, oh, od):
        """6-face corners: 6 candidate positions per placed box."""
        corners = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            corners.append((px + pw, py, pz))
            corners.append((px, py + ph, pz))
            corners.append((px, py, pz + pd))
            corners.append((px + pw, py + ph, pz))
            corners.append((px + pw, py, pz + pd))
            corners.append((px, py + ph, pz + pd))
        return corners

    def extended_corners_generator(placed, ow, oh, od):
        """8-corner generation: all 8 corners of each placed box."""
        corners = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            corners.append((px + pw, py, pz))
            corners.append((px, py + ph, pz))
            corners.append((px, py, pz + pd))
            corners.append((px + pw, py + ph, pz))
            corners.append((px + pw, py, pz + pd))
            corners.append((px, py + ph, pz + pd))
            corners.append((px + pw, py + ph, pz + pd))
            corners.append((px, py + ph, pz))
        return corners

    def shelf_generator(placed, ow, oh, od):
        """Shelf-based: try to place on top of existing shelves (z-levels)."""
        corners = [(0, 0, 0)]
        # Collect unique z-levels (shelf heights)
        z_levels = set([0])
        for _, px, py, pz, pw, ph, pd in placed:
            z_levels.add(pz + pd)
        
        # For each z-level, try x,y positions at that height
        for z in sorted(z_levels):
            if z + od > D:
                continue
            # Try x=0
            corners.append((0, 0, z))
            for _, px, py, pz, pw, ph, pd in placed:
                if pz + pd == z:
                    corners.append((px + pw, py, z))
                    corners.append((px, py + ph, z))
                    corners.append((px + pw, py + ph, z))
        
        return corners

    # Different sorting strategies
    sort_strategies = [
        lambda b: b[0]*b[1]*b[2],           # volume
        lambda b: max(b),                     # max dimension
        lambda b: sum(b),                     # sum of dimensions
        lambda b: max(b) / max(min(b), 1e-9),# elongation
        lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]),  # surface area
        lambda b: b[0]*b[1]*b[2] / max(b),   # volume per max dim
        lambda b: min(b),                     # min dimension
        lambda b: b[0]*b[1],                 # base area
    ]

    pos_generators = [corners_generator, extended_corners_generator, shelf_generator]

    best_result = None
    best_vol = 0

    # Deterministic strategies first
    for si, sort_key in enumerate(sort_strategies):
        if time.time() - start_time > DEADLINE:
            break
        for pi, pos_gen in enumerate(pos_generators):
            if time.time() - start_time > DEADLINE:
                break
            res, vol = greedy_pack(sort_key, pos_gen)
            if vol > best_vol:
                best_vol = vol
                best_result = res

    if best_result is None:
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n

    # Random restarts with diverse configurations
    rng = random.Random(42)
    restart_count = 0
    max_restarts = 200

    while time.time() - start_time < DEADLINE and restart_count < max_restarts:
        restart_count += 1

        # Create a random shuffle of box indices
        indices = list(range(n))
        rng.shuffle(indices)

        # Use a random sorting strategy
        sort_key = sort_strategies[rng.randint(0, len(sort_strategies) - 1)]
        pos_gen = pos_generators[rng.randint(0, len(pos_generators) - 1)]

        # Randomized greedy: when multiple positions are equally good, pick randomly
        indexed_boxes = sorted(indices, key=lambda i: sort_key(boxes[i]), reverse=True)

        placed = []
        total_vol = 0

        for idx in indexed_boxes:
            if time.time() - start_time > DEADLINE:
                break

            best_placement = None
            best_score = float('inf')

            for ow, oh, od in box_orientations[idx]:
                candidate_positions = pos_gen(placed, ow, oh, od)

                for cx, cy, cz in candidate_positions:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if not valid:
                        continue

                    score = cx + cy + cz
                    # Add random perturbation to break ties and explore
                    score += rng.random() * 0.05

                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)

            if best_placement:
                placed.append(best_placement)
                total_vol += best_placement[4] * best_placement[5] * best_placement[6]

        if total_vol > best_vol:
            best_vol = total_vol
            result = [(-1, 0, 0, 0, 0, 0, 0)] * n
            for idx, x, y, z, w, h, d in placed:
                result[idx] = (0, x, y, z, w, h, d)
            best_result = result

    return best_result
