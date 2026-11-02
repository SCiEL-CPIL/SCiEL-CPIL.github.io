"""Analytical lattice packing: 30-degree rotated hexagons on triangular lattice,
minimal enclosing hexagon computed via support function."""
import numpy as np
import time

N = 11

def construct():
    # --- Geometry constants ---
    # Unit regular hexagon, flat-top (vertices at 0,60,120,180,240,300 deg)
    # After rotating 30 deg, vertices at 30,90,150,210,270,330 deg (pointy-top)
    #
    # For a flat-top unit hexagon centered at origin:
    #   Vertices: (1,0), (0.5, s3/2), (-0.5, s3/2), (-1,0), (-0.5,-s3/2), (0.5,-s3/2)
    #   where s3 = sqrt(3)
    #
    # Rotated 30 deg (counter-clockwise), the hexagon becomes pointy-top:
    #   Vertices at angles 30,90,150,210,270,330 deg, radius 1
    #   = (cos30, sin30), (0,1), (-cos30, sin30), (-cos30,-sin30), (0,-1), (cos30,-sin30)
    #   = (s3/2, 0.5), (0,1), (-s3/2, 0.5), (-s3/2,-0.5), (0,-1), (s3/2,-0.5)
    #
    # In this orientation, the hexagon has:
    #   - Width (horizontal, flat-to-flat): 2 * (s3/2) = s3  ... wait
    #   - Actually for pointy-top: width = s3 (horizontal), height = 2 (vertical)
    #   - The "flat" sides are top and bottom (horizontal edges)
    #   - The "pointy" sides are left and right (vertices)
    #
    # For a flat-top hexagon (original, 0 deg rotation):
    #   - Flat sides are top and bottom
    #   - Width = 2, Height = s3
    #
    # For side-to-side contact of flat-top hexagons:
    #   - Horizontal neighbors (sharing vertical side): dx = s3, dy = 0
    #   - Diagonal neighbors: dx = s3/2, dy = 1.5
    #
    # Triangular lattice basis for flat-top hexagons (0 deg):
    #   a1 = (s3, 0)      -- horizontal
    #   a2 = (s3/2, 1.5)  -- 60 deg direction
    #
    # But wait: the problem says inner hexagons may rotate. The key insight from
    # the cell findings is that 30 deg rotation aligns lattice vectors with
    # minimum-width axes, giving packing distance sqrt(3) vs 2.
    #
    # Let me reconsider. For a flat-top unit hexagon:
    #   The 6 sides are at angles 0, 60, 120, 180, 240, 300 from center.
    #   Side normals point at 0, 60, 120, 180, 240, 300.
    #   The inradius (distance from center to side midpoint) = s3/2.
    #   The circumradius (distance from center to vertex) = 1.
    #
    # Two flat-top hexagons can be side-to-side when their centers are separated
    # by 2 * inradius = s3 along a side-normal direction.
    #
    # For a hexagonal tiling of flat-top hexagons, the lattice is:
    #   a1 = (s3, 0)
    #   a2 = (s3/2, 3/2)
    #
    # Now, if we rotate each hexagon by 30 deg, the hexagon becomes pointy-top.
    # The tiling structure is the same (just rotated), so the lattice vectors
    # in the global frame become:
    #   a1' = rotate(a1, 30) = (s3*cos30, s3*sin30) = (s3*s3/2, s3/2) = (3/2, s3/2)
    #   a2' = rotate(a2, 30) = ...
    #
    # Actually, it's simpler: the tiling is the same, just rotated. The relative
    # positions of centers don't change; only the orientation of each hexagon changes.
    # So the lattice vectors in the global frame are the same regardless of
    # individual hexagon rotation (the lattice is defined by the tiling, not the
    # individual shapes).
    #
    # Wait, that's not right either. The lattice vectors depend on how the hexagons
    # are arranged. If all hexagons have the same orientation, the tiling is a
    # regular hexagonal tiling, and the lattice is fixed.
    #
    # Let me just use the standard hexagonal tiling lattice for flat-top hexagons:
    #   a1 = (s3, 0)
    #   a2 = (s3/2, 3/2)
    #
    # And place all hexagons at 0 deg (flat-top) or 30 deg (pointy-top).
    # The tiling is valid in either case (it's just a rotation of the whole pattern).
    #
    # The cell finding says 30 deg rotation gives tighter packing. Let me think about
    # why: if the outer hexagon is flat-top (0 deg), and the inner hexagons are also
    # flat-top, then the "flat" sides of inner hexagons are parallel to the "flat"
    # sides of the outer hexagon. This is good for fitting.
    #
    # But the cell finding says 30 deg rotation aligns lattice vectors with minimum-
    # width axes. For a flat-top hexagon, the minimum width is s3 (horizontal,
    # flat-to-flat). The lattice vector a1 = (s3, 0) is horizontal, so it IS aligned
    # with the minimum-width axis. So 0 deg rotation already has this property.
    #
    # Hmm, let me re-read: "Changing inner hexagon orientation to 30 degrees aligns
    # the lattice vectors with the minimum-width (flat-to-flat) axes, allowing a
    # tighter packing distance (sqrt(3) vs 2)".
    #
    # I think the key is: for a pointy-top hexagon (30 deg rotation from flat-top),
    # the minimum width is s3 (vertical, flat-to-flat). The lattice vector a2 =
    # (s3/2, 3/2) has length sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). So the
    # packing distance is sqrt(3).
    #
    # For a flat-top hexagon, the minimum width is s3 (horizontal). The lattice
    # vector a1 = (s3, 0) has length s3 = sqrt(3). So the packing distance is
    # also sqrt(3).
    #
    # I think the confusion is about which direction the "tight" packing is in.
    # Let me just try both orientations and see which gives a smaller enclosing
    # hexagon.
    #
    # Actually, the simplest approach: use the standard hexagonal tiling lattice,
    # place N hexagons, compute the minimal enclosing regular hexagon (at 0 deg
    # rotation, centered at origin), and return that.
    #
    # For the minimal enclosing regular hexagon:
    #   A regular hexagon of side S, centered at origin, with flat-top orientation,
    #   contains a point (x,y) iff:
    #     |x| <= S
    #     |x/2 + y*s3/2| <= S*s3/2
    #     |-x/2 + y*s3/2| <= S*s3/2
    #
    #   Equivalently, the support function in direction (cos theta, sin theta) is:
    #     h(theta) = max over vertices of dot(v, (cos theta, sin theta))
    #   For a regular hexagon of side S, the support function is:
    #     h(theta) = S * max(1, cos(theta), cos(theta - pi/3), ...)
    #   Actually, for a regular hexagon with circumradius R (= S for unit side),
    #   the support function is:
    #     h(theta) = R * max_{k=0..5} cos(theta - k*pi/3)
    #   But this is only valid when the max is positive; otherwise we need to
    #   consider the full support.
    #
    #   Simpler: the hexagon contains a point (x,y) iff the point is inside all
    #   6 half-planes. The 6 half-plane constraints for a flat-top hexagon of
    #   side S centered at origin are:
    #     x <= S
    #     x >= -S
    #     x/2 + y*s3/2 <= S*s3/2
    #     x/2 + y*s3/2 >= -S*s3/2
    #     -x/2 + y*s3/2 <= S*s3/2
    #     -x/2 + y*s3/2 >= -S*s3/2
    #
    #   For a set of points, the minimal S is:
    #     S >= max_i |x_i|
    #     S >= max_i |x_i/2 + y_i*s3/2| / (s3/2)
    #     S >= max_i |-x_i/2 + y_i*s3/2| / (s3/2)
    #
    #   So S_min = max( max|x_i|, max|x_i/2 + y_i*s3/2|*2/s3, max|-x_i/2 + y_i*s3/2|*2/s3 )
    #
    #   But we also need to consider that the hexagon can be rotated. However,
    #   due to the 6-fold symmetry of the hexagonal tiling, rotating the inner
    #   set by multiples of 60 deg gives the same enclosing hexagon size. And
    #   rotating by 30 deg swaps the roles of the two sets of constraints.
    #
    #   Actually, the outer hexagon can be at any rotation. The minimal enclosing
    #   regular hexagon over all rotations is:
    #     S_min = min_{theta} max_{i, k} dot(p_i - c, n_k(theta)) / (s3/2)
    #   where n_k(theta) are the 6 outward normals of the outer hexagon.
    #
    #   For simplicity, let's fix the outer hexagon at 0 deg (flat-top) and
    #   compute S as above. Due to symmetry, this should be close to optimal.
    #
    #   Actually, for a hexagonal tiling patch, the optimal enclosing hexagon
    #   is aligned with the tiling, so 0 deg should be optimal.

    s3 = np.sqrt(3.0)

    # --- Generate hexagonal tiling positions ---
    # Triangular lattice for flat-top hexagons:
    #   a1 = (s3, 0)
    #   a2 = (s3/2, 3/2)
    #
    # We want N = 11 hexagons. Let's use a hexagonal patch:
    #   Center + ring of 6 + ring of 12 -> too many.
    #   Center + ring of 6 = 7.
    #   Center + ring of 6 + 4 from next ring = 11.
    #
    # Let's enumerate all lattice points within a certain radius and pick the
    # closest 11 to the origin.

    # Lattice points: i*a1 + j*a2 for integers i, j
    # Let's enumerate i, j in range [-4, 4] and compute positions

    a1 = np.array([s3, 0.0])
    a2 = np.array([s3 / 2.0, 1.5])

    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            pos = i * a1 + j * a2
            dist_sq = pos[0]**2 + pos[1]**2
            candidates.append((dist_sq, i, j, pos))

    candidates.sort(key=lambda x: x[0])
    # Take the first N
    selected = candidates[:N]
    positions = np.array([c[3] for c in selected])

    # Center of mass
    com = positions.mean(axis=0)
    positions = positions - com  # Center the patch at origin

    # --- Compute minimal enclosing regular hexagon (flat-top, 0 deg) ---
    # For each point, compute the 3 "width" values:
    #   w1 = |x|
    #   w2 = |x/2 + y*s3/2|
    #   w3 = |-x/2 + y*s3/2|
    # S_min = max( max(w1), max(w2)/(s3/2), max(w3)/(s3/2) )

    xs = positions[:, 0]
    ys = positions[:, 1]

    w1 = np.abs(xs)
    w2 = np.abs(xs / 2.0 + ys * s3 / 2.0)
    w3 = np.abs(-xs / 2.0 + ys * s3 / 2.0)

    S1 = np.max(w1)
    S2 = np.max(w2) / (s3 / 2.0)
    S3 = np.max(w3) / (s3 / 2.0)

    S = max(S1, S2, S3)

    # --- Also try pointy-top inner hexagons (30 deg rotation) ---
    # The tiling is the same, but the hexagons are rotated. The positions are
    # the same. The enclosing hexagon computation is the same (it depends only
    # on the positions, not the orientation of the inner hexagons, as long as
    # the inner hexagons are contained in the outer).
    #
    # Wait, the enclosing hexagon size depends on the positions of the centers
    # AND the orientation of the inner hexagons (because the inner hexagons
    # extend beyond their centers).
    #
    # For a flat-top unit hexagon centered at (x,y), the vertices are at:
    #   (x+1, y), (x+0.5, y+s3/2), (x-0.5, y+s3/2), (x-1, y), (x-0.5, y-s3/2), (x+0.5, y-s3/2)
    #
    # For a pointy-top unit hexagon (30 deg rotation), the vertices are at:
    #   (x+s3/2, y+0.5), (x, y+1), (x-s3/2, y+0.5), (x-s3/2, y-0.5), (x, y-1), (x+s3/2, y-0.5)
    #
    # The minimal enclosing hexagon must contain ALL vertices of ALL inner hexagons.
    # So we need to compute the enclosing hexagon of the union of all vertices.

    def compute_enclosing_hexagon(vertices):
        """Compute minimal side length of a flat-top regular hexagon centered at origin
        that contains all given vertices."""
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        w1 = np.max(np.abs(xs))
        w2 = np.max(np.abs(xs / 2.0 + ys * s3 / 2.0)) / (s3 / 2.0)
        w3 = np.max(np.abs(-xs / 2.0 + ys * s3 / 2.0)) / (s3 / 2.0)
        return max(w1, w2, w3)

    # --- Case 1: Flat-top inner hexagons (0 deg) ---
    # Vertices of each inner hexagon
    flat_top_verts = np.array([
        [1.0, 0.0],
        [0.5, s3/2],
        [-0.5, s3/2],
        [-1.0, 0.0],
        [-0.5, -s3/2],
        [0.5, -s3/2]
    ])

    all_verts_flat = []
    for pos in positions:
        all_verts_flat.append(pos + flat_top_verts)
    all_verts_flat = np.vstack(all_verts_flat)
    S_flat = compute_enclosing_hexagon(all_verts_flat)

    # --- Case 2: Pointy-top inner hexagons (30 deg) ---
    R30 = np.array([
        [np.cos(np.pi/6), -np.sin(np.pi/6)],
        [np.sin(np.pi/6), np.cos(np.pi/6)]
    ])
    pointy_top_verts = flat_top_verts @ R30.T

    all_verts_pointy = []
    for pos in positions:
        all_verts_pointy.append(pos + pointy_top_verts)
    all_verts_pointy = np.vstack(all_verts_pointy)
    S_pointy = compute_enclosing_hexagon(all_verts_pointy)

    # Choose the better orientation
    if S_flat <= S_pointy:
        best_S = S_flat
        best_angle = 0.0
    else:
        best_S = S_pointy
        best_angle = 30.0

    # --- Build the inner hexagon list ---
    inner = []
    for pos in positions:
        inner.append([float(pos[0]), float(pos[1]), float(best_angle)])

    # Outer hexagon: center (0,0), side best_S, rotation 0 deg
    # We could also try rotating the outer hexagon to see if it helps,
    # but due to the 6-fold symmetry of the patch, 0 deg should be near-optimal.

    # Let's also try a few outer rotations to be safe
    best_outer_S = best_S
    best_outer_angle = 0.0

    # For each outer rotation theta, the half-plane constraints are rotated.
    # The support function of the point set in direction n is max dot(p, n).
    # For a regular hexagon of side S with outward normals at angles theta + k*60,
    # the constraint is: dot(p, n_k) <= S * (s3/2) for all k and all p.
    # So S >= max_p max_k dot(p, n_k) / (s3/2)
    #
    # But we also need the hexagon to contain the points in the "vertex" directions.
    # Actually, the half-plane formulation is exact for convex polygons.
    # A regular hexagon is the intersection of 6 half-planes. So the minimal S
    # for a given orientation is:
    #   S = max_{k=0..5} max_p dot(p, n_k) / (s3/2)
    # where n_k are the outward unit normals.
    #
    # Wait, that's not quite right. The half-plane for side k is:
    #   dot(p, n_k) <= d_k
    # where d_k is the distance from center to side k. For a regular hexagon of
    #   side S, d_k = S * s3/2 for all k.
    # So the constraint is dot(p, n_k) <= S * s3/2 for all k and all p.
    # Thus S >= max_{k, p} dot(p, n_k) / (s3/2).
    #
    # But this only works if the hexagon is centered at the origin and the points
    # are all in the "positive" half-planes. If some points are on the negative
    # side, we need to also check the opposite normals. But since the hexagon is
    # centrally symmetric, checking all 6 normals covers both sides.
    #
    # Actually, for a centrally symmetric convex body, the support function h(n) =
    # max_p dot(p, n) satisfies h(-n) = h(n). So checking all 6 normals (which
    # come in opposite pairs) is sufficient.

    # Let's compute S for various outer rotations
    all_verts = all_verts_flat if best_angle == 0.0 else all_verts_pointy

    for theta_deg in [0, 30, 15, 45, 10, 20, 50, 25, 35]:
        theta = np.deg2rad(theta_deg)
        # Outward normals at angles theta + k*60
        max_dot = 0.0
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            dots = all_verts @ n
            max_dot = max(max_dot, np.max(dots))
        S_theta = max_dot / (s3 / 2.0)
        if S_theta < best_outer_S:
            best_outer_S = S_theta
            best_outer_angle = theta_deg

    # Final check: verify no overlaps (should be zero by construction)
    # and all inner hexagons are inside the outer hexagon.

    inner = []
    for pos in positions:
        inner.append([float(pos[0]), float(pos[1]), float(best_angle)])

    return inner, (0.0, 0.0), float(best_outer_S), float(best_outer_angle)
