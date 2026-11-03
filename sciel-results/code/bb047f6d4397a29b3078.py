    # Inside the loop over candidates
    pts = axial_to_cartesian(axial)
    S, beta, center = optimize_shape(pts)
    # Check if this specific result is strictly disjoint
    angles_deg_check = [math.degrees(beta)] * len(pts)
    if check_all_disjoint(pts, angles_deg_check, tol=1e-10):
         if S < best_strict_S:
             best_strict_S = S
             best_strict_pts = pts
             ...
    else:
         # It's invalid. Maybe the lattice version was better?
         # optimize_shape might have returned a relaxed invalid one.
         # We should evaluate the lattice version's S separately?
         # Or just keep the previous best_strict.
         pass
    