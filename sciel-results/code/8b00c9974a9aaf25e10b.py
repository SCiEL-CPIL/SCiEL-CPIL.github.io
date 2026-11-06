"""3D Container Loading using maximal-empty-boxes (free-space) packer with rotation."""

def _orientations(w, h, d):
    """Return all 6 unique axis rotations of (w, h, d)."""
    rots = set()
    rots.add((w, h, d))
    rots.add((w, d, h))
    rots.add((h, w, d))
    rots.add((h, d, w))
    rots.add((d, w, h))
    rots.add((d, h, w))
    return list(rots)

def _fits(box, free):
    """Check if box (w,h,d) fits in free space (fx,fy,fz, fw,fh,fd)."""
    w, h, d = box
    fx, fy, fz, fw, fh, fd = free
    return (w <= fw and h <= fh and d <= fd)

def _split_free(free, box_pos, box_size):
    """Split a free space by removing a placed box. Returns list of new free spaces."""
    fx, fy, fz, fw, fh, fd = free
    bx, by, bz, bw, bh, bd = box_pos
    # The placed box occupies [bx, bx+bw) x [by, by+bh) x [bz, bz+bd)
    # Within the free space, the box is at local coords:
    lx, ly, lz = bx - fx, by - fy, bz - fz
    new_spaces = []
    # Below the box (in z)
    if lz > 0:
        new_spaces.append((fx, fy, fz, fw, fh, lz))
    # Above the box (in z)
    if lz + bd < fd:
        new_spaces.append((fx, fy, fz + lz + bd, fw, fh, fd - (lz + bd)))
    # In the same z-slab as the box, split in y
    if lz <= 0 and lz + bd >= fd:
        # Box spans full height of this free space in z
        pass
    else:
        # There is a z-slab where the box exists: [lz, lz+bd)
        z0, z1 = lz, lz + bd
        # Left of box (in x) within that z-slab
        if lx > 0:
            new_spaces.append((fx, fy, fz + z0, lx, fh, z1 - z0))
        # Right of box (in x) within that z-slab
        if lx + bw < fw:
            new_spaces.append((fx + lx + bw, fy, fz + z0, fw - (lx + bw), fh, z1 - z0))
        # In the same x-slab as the box, split in y
        if ly > 0:
            new_spaces.append((fx + lx, fy, fz + z0, bw, ly, z1 - z0))
        if ly + bh < fh:
            new_spaces.append((fx + lx, fy + ly + bh, bw, fh - (ly + bh), z1 - z0))
    # Deduplicate and remove empty/contained spaces
    # Remove spaces with zero volume
    new_spaces = [s for s in new_spaces if s[3] > 0 and s[4] > 0 and s[5] > 0]
    # Remove spaces that are contained in another
    def contains(a, b):
        # a contains b?
        ax, ay, az, aw, ah, ad = a
        bx, by, bz, bw, bh, bd = b
        return (ax <= bx and ay <= by and az <= bz and
                ax + aw >= bx + bw and ay + ah >= by + bh and az + ad >= bz + bd)
    kept = []
    for i, s in enumerate(new_spaces):
        contained = False
        for j, t in enumerate(new_spaces):
            if i != j and contains(t, s):
                contained = True
                break
        if not contained:
            kept.append(s)
    return kept

def load(boxes, container):
    W, H, D = container
    # Sort boxes: largest volume first, tie-break by largest dimension
    indexed = sorted(enumerate(boxes), key=lambda t: (-(t[1][0]*t[1][1]*t[1][2]), -max(t[1])), reverse=False)
    # Actually we want largest first: sort descending by volume
    indexed = sorted(enumerate(boxes), key=lambda t: (t[1][0]*t[1][1]*t[1][2]), reverse=True)
    
    placements = [None] * len(boxes)
    free_spaces = [(0, 0, 0, W, H, D)]
    
    for idx, (w, h, d) in indexed:
        placed = False
        best_key = None
        best_choice = None
        for ow, oh, od in _orientations(w, h, d):
            for fi, free in enumerate(free_spaces):
                if _fits((ow, oh, od), free):
                    fx, fy, fz, fw, fh, fd = free
                    # Place at bottom-left-farthest within this free space
                    bx, by, bz = fx, fy, fz
                    key = (bz, by, bx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_choice = (idx, ow, oh, od, bx, by, bz, fi)
        if best_choice is not None:
            idx, ow, oh, od, bx, by, bz, fi = best_choice
            placements[idx] = (0, bx, by, bz, ow, oh, od)
            # Update free spaces
            new_spaces = _split_free(free_spaces[fi], (bx, by, bz, ow, oh, od), (ow, oh, od))
            free_spaces = free_spaces[:fi] + free_spaces[fi+1:] + new_spaces
            # Remove any free spaces that are now contained in another (cleanup)
            def contains(a, b):
                ax, ay, az, aw, ah, ad = a
                bx2, by2, bz2, bw2, bh2, bd2 = b
                return (ax <= bx2 and ay <= by2 and az <= bz2 and
                        ax + aw >= bx2 + bw2 and ay + ah >= by2 + bh2 and az + ad >= bz2 + bd2)
            kept = []
            for i, s in enumerate(free_spaces):
                contained = False
                for j, t in enumerate(free_spaces):
                    if i != j and contains(t, s):
                        contained = True
                        break
                if not contained:
                    kept.append(s)
            free_spaces = kept
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    return placements
