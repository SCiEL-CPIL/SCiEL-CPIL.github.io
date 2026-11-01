"""Fresh design: analytical 6x2 grid packing of 12 unit flat-top hexagons
inside a pointy-top regular hexagon of side length 9."""
import math
import time

N = 12
DEADLINE = 135.0  # safety margin before 150s wall-clock kill

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()

    # 6 columns x 2 rows of flat-top unit hexagons.
    # Flat-top hexagon: width 2, height sqrt(3).
    # Horizontal pitch 3 (touching on flat edges), vertical pitch sqrt(3).
    cols = 6
    rows = 2
    x0 = -(cols - 1) / 2.0 * 3.0  # -7.5
    y0 = -(rows - 1) / 2.0 * math.sqrt(3)  # -sqrt(3)/2

    inner = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * 3.0
            y = y0 + r * math.sqrt(3)
            inner.append([x, y, 0.0])  # angle 0 = flat-top orientation

    # Outer: pointy-top regular hexagon, center origin, side length 9.
    # Pointy-top rotation in this convention: 90 degrees.
    # Side 9 -> vertical extent 18 (covers grid height sqrt(3) ~ 1.732),
    #          horizontal extent 9*sqrt(3) ~ 15.588 (covers grid width 18? NO).
    # Wait: grid width is 18 (x from -9 to 9). Need 9*sqrt(3) >= 18 -> sqrt(3)>=2 -> false.
    # So we must check: does pointy-top with side 9 cover x in [-9,9]?
    # Pointy-top hexagon: vertices at angles 30,90,150,210,270,330 deg * side.
    # At 30 deg: (S*cos30, S*sin30) = (S*sqrt(3)/2, S/2). Max x = S*sqrt(3)/2.
    # For S=9: max x = 9*sqrt(3)/2 ~ 7.794 < 9. NOT enough.

    # So pointy-top side 9 does NOT contain the 18-wide grid.
    # Re-evaluate: flat-top outer hexagon has horizontal extent 2*S.
    # Need 2*S >= 18 -> S >= 9. And vertical extent S*sqrt(3) >= sqrt(3) (grid height) -> S>=1.
    # Flat-top outer side 9: horizontal 18 (exactly fits), vertical 9*sqrt(3) ~ 15.588.
    # Grid height sqrt(3) ~ 1.732 << 15.588. So flat-top outer side 9 works!

    # Flat-top outer hexagon: rotation 0 degrees, side 9.
    outer_center = (0.0, 0.0)
    outer_side = 9.0
    outer_angle = 0.0

    if time.time() - t0 > DEADLINE:
        return inner, outer_center, outer_side, outer_angle

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
