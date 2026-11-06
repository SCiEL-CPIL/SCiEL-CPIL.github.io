def _place_blf(boxes, container, order, box_rotations, deadline, W, H, D):
    n = len(boxes)
    # placed_np will be a numpy array of shape (k, 6)
    placed_np = np.empty((0, 6))
    result = [None] * n

    for i in order:
        if time.time() > deadline:
            break
        bw, bh, bd = boxes[i]
        rots = box_rotations[i]

        # Generate candidate coordinates
        xs = [0]
        ys = [0]
        zs = [0]
        if len(placed_np) > 0:
            px, py, pz, pw, ph, pd = placed_np.T
            xs.extend((px + pw).tolist())
            ys.extend((py + ph).tolist())
            zs.extend((pz + pd).tolist())
        
        xs = sorted(set(x for x in xs if x <= W))
        ys = sorted(set(y for y in ys if y <= H))
        zs = sorted(set(z for z in zs if z <= D))

        best_pos = None
        found = False
        # BLF: scan z asc, y asc, x asc
        for z in zs:
            if found: break
            for y in ys:
                if found: break
                for x in xs:
                    if found: break
                    for (rw, rh, rd) in rots:
                        if x + rw > W or y + rh > H or z + rd > D:
                            continue
                        if _check_overlap_np(placed_np, x, y, z, rw, rh, rd):
                            continue
                        best_pos = (x, y, z, rw, rh, rd)
                        found = True
                        break
        
        if best_pos is not None:
            x, y, z, rw, rh, rd = best_pos
            # Append to numpy array
            new_box = np.array([[x, y, z, rw, rh, rd]])
            placed_np = np.vstack((placed_np, new_box))
            result[i] = (0, x, y, z, rw, rh, rd)
        else:
            result[i] = (-1, 0, 0, 0, bw, bh, bd)

    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return result, placed_np
