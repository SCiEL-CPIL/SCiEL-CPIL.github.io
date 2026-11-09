"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Strategy: Place 11 points on a single line and 1 point off the line.
    This configuration results in exactly 11 ordinary lines (each connecting the off-point to one of the 11 line-points).
    The 11 points on the line form 1 rich line (containing 11 points), so no ordinary lines arise from pairs within that set.
    The off-point forms 11 distinct lines with each of the 11 points. None of these lines contain a 3rd point 
    because the 11 points are collinear on a different line, and no other points exist.
    Total ordinary lines = 11.
    """
    points = []
    # 11 points on the x-axis: (0,0) to (10,0)
    for i in range(11):
        points.append([i, 0])
    # 1 point off the line: (0,1)
    points.append([0, 1])
    return points
# EVOLVE-BLOCK-END
