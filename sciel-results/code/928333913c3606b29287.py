<<<<<<< SEARCH
def _inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, eps=1e-9):
    """Check all inner hexagons are inside the outer hexagon."""
    for hex_verts in inner_hexes:
        for v in hex_verts:
            if not _point_in_hex(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle, eps):
                return False
    return True
=======
def _inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, eps=1e-9):
    """Check all inner hexagons are inside the outer hexagon."""
    for hex_verts in inner_hexes:
        for v in hex_verts:
            # Outer hexagon is defined by normals at outer_angle, so is_flat_top=True
            if not _point_in_hex(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle, is_flat_top=True):
                return False
    return True
>>>>>>> REPLACE
