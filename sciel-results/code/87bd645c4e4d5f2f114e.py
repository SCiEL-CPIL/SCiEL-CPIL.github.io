<<<<<<< SEARCH
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2, 1.5)
=======
    sqrt3 = math.sqrt(3)
    # Scale spacing by 1.01 to ensure strict disjointness (gap between hexagons)
    # This avoids floating-point precision issues where touching edges are seen as overlapping
    scale = 1.01
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
>>>>>>> REPLACE
