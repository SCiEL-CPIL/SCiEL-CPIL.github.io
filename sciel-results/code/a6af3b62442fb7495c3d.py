import time
import random

DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _pack_shelves(boxes, container, order, rng):
    """
    Shelf-based 3D bin packing.
    Each bin is a list of shelves. Each shelf is a list of boxes packed in a 2D strip
    (x is horizontal, z is depth). Shelf height is determined by the tallest box on it.
    Boxes are placed in the order given by 'order'.
    """
    W, H, D = container
    n = len(boxes)
    bins = []  # list of bins, each bin is a list of shelves
    bin_vols = []  # volume in each bin

    # We will assign each box to a bin and a shelf.
    # For each box, we try to fit it into an existing shelf in an existing bin,
    # or create a new shelf in an existing bin, or create a new bin.

    # To make it faster, we can precompute some things, but for 40 boxes it's fine.

    # We'll keep track of for each bin:
    # - shelves: list of (shelf_height, current_x, boxes_on_shelf)
    # - current_z: the y-coordinate where the next shelf will start

    for idx in order:
        w, h, d = boxes[idx]
        box_vol = w * h * d

        if not bins:
            # Create first bin, first shelf
            bins.append([[(h, w, [(idx, 0, 0, w, h, d)])]])
            bin_vols.append(box_vol)
            continue

        best_bin_idx = -1
        best_shelf_idx = -1
        best_x = 0
        best_score = float('inf')

        for bi in range(len(bins)):
            shelves = bins[bi]
            current_z = sum(sh[0] for sh in shelves)

            # Try to fit in an existing shelf
            for si in range(len(shelves)):
                shelf_h, shelf_x, shelf_boxes = shelves[si]
                # Can the box fit in this shelf?
                # Height constraint: h <= shelf_h (since shelf height is fixed by first box)
                # Actually, in shelf packing, the shelf height is the max height of boxes on it.
                # If we add a box with height h, and h > shelf_h, we need to increase shelf height.
                # But that would change the z-position of all subsequent shelves.
                # To keep it simple, we only allow adding a box if h <= shelf_h.
                if h > shelf_h:
                    continue
                # Width constraint: shelf_x + w <= W
                if shelf_x + w > W:
                    continue
                # Depth constraint: d <= D (since we place at z=0 for simplicity in this variant)
                if d > D:
                    continue

                # Compute a score for placing here
                # Prefer bins that are already full, shelves that are close to full
                # We want to minimize the "waste" created.
                # Score: current_x + w (how much of the shelf is used) - we want to minimize this?
                # No, we want to maximize utilization. So minimize the remaining space.
                # Actually, we want to pack tightly. Let's use the "best fit" idea:
                # Place in the shelf where shelf_x + w is smallest (tightest fit).
                score = shelf_x + w
                # Also consider the bin's current fill to prefer filling one bin
                # Add a small penalty for using a new bin if we can use an old one?
                # But we're iterating over bins, so we'll compare across bins.
                # Let's add a small bonus for bins with higher volume (to encourage filling)
                score -= bin_vols[bi] * 0.001

                if score < best_score:
                    best_score = score
                    best_bin_idx = bi
                    best_shelf_idx = si
                    best_x = shelf_x

            # If no existing shelf works, consider creating a new shelf in this bin
            # New shelf starts at current_z
            if current_z + h <= H and w <= W and d <= D:
                # Score for new shelf: prefer bins that are already full
                # The "cost" of a new shelf is its height.
                # We want to minimize the total height used.
                # Score: current_z + h (new total height)
                score = current_z + h
                # Bonus for filling existing bins
                score -= bin_vols[bi] * 0.001

                if score < best_score:
                    best_score = score
                    best_bin_idx = bi
                    best_shelf_idx = len(shelves)  # new shelf
                    best_x = 0

        if best_bin_idx == -1:
            # No fit in any bin, create new bin
            bins.append([[(h, w, [(idx, 0, 0, w, h, d)])]])
            bin_vols.append(box_vol)
        else:
            if best_shelf_idx == len(bins[best_bin_idx]):
                # Create new shelf
                current_z = sum(sh[0] for sh in bins[best_bin_idx])
                bins[best_bin_idx].append([(h, w, [(idx, 0, current_z, w, h, d)])])
                bin_vols[best_bin_idx] += box_vol
            else:
                # Add to existing shelf
                shelf = bins[best_bin_idx][best_shelf_idx]
                shelf[1] += w  # update current_x
                shelf[2].append((idx, best_x, 0, w, h, d))
                bin_vols[best_bin_idx] += box_vol

    return bins


def _score(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)

    if B == 0:
        return 0.0

    last_bin = bins[-1]
    if not last_bin:
        return B

    last_vol = 0
    for shelf in last_bin:
        for (idx, x, y, z, w, h, d) in shelf[2]:
            last_vol += w * h * d

    fill_ratio = last_vol / C if C > 0 else 0.0
    return B + 0.1 * (1.0 - fill_ratio)


def _generate_orderings(boxes, n, rng):
    """Generate diverse orderings of box indices."""
    idx = list(range(n))

    # 1. Volume Descending
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    # 2. Max Dimension Descending
    yield sorted(idx, key=lambda i: max(boxes[i]), reverse=True)

    # 3. Height Descending (since shelf height is determined by tallest box)
    yield sorted(idx, key=lambda i: boxes[i][1], reverse=True)

    # 4. Width Descending
    yield sorted(idx, key=lambda i: boxes[i][0], reverse=True)

    # 5-100. Random Shuffles
    for _ in range(95):
        s = idx[:]
        rng.shuffle(s)
        yield s


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    best_bins = None
    best_score = float('inf')
    t_start = time.time()
    rng = random.Random(42)

    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break

        bins = _pack_shelves(boxes, container, order, rng)
        s = _score(bins, container)

        if s < best_score:
            best_score = s
            best_bins = bins

    if best_bins is None:
        # Fallback: one box per bin
        best_bins = []
        for i in range(n):
            w, h, d = boxes[i]
            best_bins.append([[(h, w, [(i, 0, 0, w, h, d)])]])
        best_score = _score(best_bins, container)

    # Build the result in original box order
    placement_map = {}
    for bi, bin_content in enumerate(best_bins):
        for shelf in bin_content:
            for (idx, x, y, z, w, h, d) in shelf[2]:
                placement_map[idx] = (bi, x, y, z, w, h, d)

    result = []
    for i in range(n):
        if i in placement_map:
            result.append(placement_map[i])
        else:
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))

    return result
