"""Rewritten program: two parallel lines of 15 points each to maximize equal-area triangle multiplicity."""
# EVOLVE-BLOCK-START
def construct():
    pts = []
    # 15 points on y=0
    for x in range(15):
        pts.append((x, 0))
    # 15 points on y=1
    for x in range(15):
        pts.append((x, 1))
    return pts
# EVOLVE-BLOCK-END
