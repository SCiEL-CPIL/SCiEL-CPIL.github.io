<<<<<<< SEARCH
def _project_inside(points):
    """Project points inside the triangle."""
    for i in range(N):
        x, y = points[i]
        if y < 0:
            y = 0.001
        elif y > S3H:
            y = S3H - 0.001
        if x < 0:
            x = 0.001
        elif x > 1:
            x = 0.999
        if x > y / np.sqrt(3):
            x = max(0.001, y / np.sqrt(3) - 0.001)
        if x < 1 - y / np.sqrt(3):
            x = min(0.999, 1 - y / np.sqrt(3) + 0.001)
        points[i] = [x, y]
    return points
=======
def _project_point_inside(x, y):
    """Project a single point (x,y) to the nearest point inside the triangle."""
    # Use geometric projection onto the half-planes defining the triangle
    # Triangle: (0,0), (1,0), (0.5, S3H)
    # Constraints:
    # 1. y >= 0
    # 2. y <= sqrt(3)*x  =>  sqrt(3)*x - y >= 0
    # 3. y <= sqrt(3)*(1-x) => sqrt(3)*x + y - sqrt(3) <= 0

    # If y < 0, project to y=0
    if y < 0:
        return x, 0.0

    # Check left edge: sqrt(3)*x - y >= 0
    val_left = np.sqrt(3) * x - y
    if val_left < 0:
        # Project onto line sqrt(3)*x - y = 0
        # Normal vector n = (sqrt(3), -1), norm^2 = 4
        # Distance factor d = val_left / 4
        d = val_left / 4.0
        x_new = x - d * np.sqrt(3)
        y_new = y - d * (-1)
        # Check if this projection keeps y >= 0 (it should if we are in the relevant region, but verify)
        if y_new < 0:
            # If projection goes below, the closest point is likely the vertex (0,0)
            return 0.0, 0.0
        return x_new, y_new

    # Check right edge: sqrt(3)*x + y - sqrt(3) <= 0
    val_right = np.sqrt(3) * x + y - np.sqrt(3)
    if val_right > 0:
        # Project onto line sqrt(3)*x + y - sqrt(3) = 0
        # Normal vector n = (sqrt(3), 1), norm^2 = 4
        # Distance factor d = val_right / 4
        d = val_right / 4.0
        x_new = x - d * np.sqrt(3)
        y_new = y - d * 1
        # Check if this projection keeps y >= 0
        if y_new < 0:
            # If projection goes below, the closest point is likely the vertex (1,0)
            return 1.0, 0.0
        return x_new, y_new

    # If we are here, y >= 0 and we are between the side lines.
    # We also need to ensure we are not above the top vertex?
    # The side lines intersect at (0.5, S3H).
    # If a point is "above" the triangle, it would violate both side constraints?
    # No, if it's above, e.g., (0.5, S3H + 1), 
    # val_left = sqrt(3)*0.5 - (S3H+1) = S3H - S3H - 1 = -1 < 0.
    # So it would have been caught by the left edge check.
    # Thus, if we pass both side checks and y>=0, we are inside.
    
    return x, y

def _project_inside(points):
    """Project points inside the triangle."""
    for i in range(N):
        x, y = points[i]
        nx, ny = _project_point_inside(x, y)
        # Add a tiny epsilon inward to avoid boundary issues in evaluator
        # Simple inward shift towards centroid (0.5, S3H/3)
        cx, cy = 0.5, S3H / 3.0
        dx, dy = cx - nx, cy - ny
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 1e-9:
            eps = 1e-5
            nx += (dx / dist) * eps
            ny += (dy / dist) * eps
        points[i] = [nx, ny]
    return points
>>>>>>> REPLACE

I will also add a final safety check in `construct` to ensure the returned points are valid.

