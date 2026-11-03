"""Program for unit regular hexagons inside a regular hexagon (honeycomb packing)."""
N = 11

# EVOLVE-BLOCK-START
def construct():
    # Honeycomb (hexagonal) cluster of 11 unit hexagons on a hex lattice.
    # Lattice coordinates (i, j) with |i|<=2, |j|<=2, |i+j|<=2 give exactly 11 cells:
    #   center + 6 ring-1 + 6 ring-2 (the 6 cells with max coordinate 2).
    # Spacing 2.0 (unit hex circumradius*2) makes them just touch.
    # Pointy orientation: basis vectors at 0 deg and 60 deg.
    inner = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if abs(i + j) <= 2:
                x = 2.0 * i
                y = 2.0 * j
                inner.append([x, y, 0.0])
    # The 11-hex cluster is bounded by a regular hexagon of side 3 (3 unit edges per side),
    # centered at the origin, rotation 0.
    return inner, (0.0, 0.0), 3.0, 0.0
# EVOLVE-BLOCK-END
