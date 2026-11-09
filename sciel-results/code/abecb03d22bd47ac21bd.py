def construct():
    deadline = time.time() + 142
    n = 32

    hex_patch = _build_hexagonal_patch(n)
    e_hex = _count_edges(hex_patch)

    if e_hex == 76:
        return [[a, b] for (a, b) in hex_patch]

    order = _spiral_order(12)
    spiral_pts = order[:n]
    e_spiral = _count_edges(spiral_pts)

    if e_spiral == 76:
        return [[a, b] for (a, b) in spiral_pts]

    if e_hex >= e_spiral:
        best = hex_patch
        best_e = e_hex
    else:
        best = spiral_pts
        best_e = e_spiral

    # ... rest of the search
