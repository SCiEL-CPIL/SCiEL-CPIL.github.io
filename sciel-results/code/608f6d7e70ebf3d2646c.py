<<<<<<< SEARCH
    # Randomized Restart with Free Placement
    # Since N=60 is small, we can afford many random restarts
    rng = random.Random(42)
    random_trials = 0
    max_random_trials = 200  # Increased since overlap check is faster
    
    while time.time() < deadline - 5.0:
=======
    # Local search polish: try to improve the best placement by moving boxes
    def local_search(placements, total_vol, max_iters=50):
        """Try to improve placement by repositioning boxes."""
        if time.time() > deadline - 2.0:
            return placements, total_vol

        # Extract placed boxes
        placed_boxes = []
        for i in range(n):
            if placements[i][0] == 0:
                _, x, y, z, w, h, d = placements[i]
                placed_boxes.append((i, x, y, z, w, h, d))

        if not placed_boxes:
            return placements, total_vol

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2.0:
            improved = False
            iters += 1

            for idx, bx, by, bz, bw, bh, bd in placed_boxes:
                if time.time() > deadline - 2.0:
                    break

                # Try to move this box to a better position
                # Remove it temporarily
                placed_boxes.remove((idx, bx, by, bz, bw, bh, bd))

                # Try all orientations
                best_pos = None
                best_orient = None
                best_score = None

                for (ow, oh, od) in box_orientations[idx]:
                    # Find best position for this orientation
                    candidates = [(0, 0, 0)]
                    for (pi, px, py, pz, pw, ph, pd) in placed_boxes:
                        candidates.append((px + pw, py, pz))
                        candidates.append((px, py + ph, pz))
                        candidates.append((px, py, pz + pd))
                        candidates.append((px + pw, py + ph, pz))
                        candidates.append((px + pw, py, pz + pd))
                        candidates.append((px, py + ph, pz + pd))
                        candidates.append((px + pw, py + ph, pz + pd))

                    seen = set()
                    valid_cands = []
                    for (cx, cy, cz) in candidates:
                        if (cx, cy, cz) in seen:
                            continue
                        seen.add((cx, cy, cz))
                        if cx >= 0 and cy >= 0 and cz >= 0:
                            if cx + ow <= W and cy + oh <= H and cz + od <= D:
                                # Check overlap
                                overlap = False
                                for (pi, px, py, pz, pw, ph, pd) in placed_boxes:
                                    if cx < px + pw and px < cx + ow and \
                                       cy < py + ph and py < cy + oh and \
                                       cz < pz + pd and pz < cz + od:
                                        overlap = True
                                        break
                                if not overlap:
                                    valid_cands.append((cx, cy, cz))

                    for (cx, cy, cz) in valid_cands:
                        score = (cz, cy, cx)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)

                # Re-add the box (possibly at new position)
                if best_pos is not None:
                    # Check if this is actually better than current
                    if (bx, by, bz) != best_pos or (bw, bh, bd) != best_orient:
                        placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                          best_orient[0], best_orient[1], best_orient[2])
                        placed_boxes.append((idx, best_pos[0], best_pos[1], best_pos[2],
                                            best_orient[0], best_orient[1], best_orient[2]))
                        improved = True
                    else:
                        placed_boxes.append((idx, bx, by, bz, bw, bh, bd))
                else:
                    # Couldn't reposition, keep original
                    placed_boxes.append((idx, bx, by, bz, bw, bh, bd))

        return placements, total_vol

    # Apply local search to best placement
    if best_placements is not None:
        best_placements, _ = local_search(best_placements, best_vol)

    # Randomized Restart with Free Placement
    # Since N=60 is small, we can afford many random restarts
    rng = random.Random(42)
    random_trials = 0
    max_random_trials = 200  # Increased since overlap check is faster
    
    while time.time() < deadline - 10.0:
>>>>>>> REPLACE

