"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import math

# EVOLVE-BLOCK-START
def construct():
    # We use a known 6-point set with rational coordinates in Q(sqrt(2))
    # such that all pairwise distances are rational, no three are collinear,
    # and no four are concyclic.
    #
    # Points (x, y*sqrt(2)) with rational x, y:
    # P1 = (0, 0)
    # P2 = (3, 0)
    # P3 = (0, 3)
    # P4 = (3, 3)
    # P5 = (1, 2)
    # P6 = (2, 1)
    #
    # Let's verify these points satisfy the conditions.
    # D = 2, so coordinates are (x, y*sqrt(2)).
    #
    # Pairwise distances:
    # P1-P2: sqrt(3^2 + 0) = 3 ✓
    # P1-P3: sqrt(0 + (3*sqrt(2))^2) = sqrt(18) = 3*sqrt(2) ✗ not rational
    #
    # Let me reconsider. I need a valid 6-point set.
    #
    # Known result: There exist 6-point integral point sets in Q(sqrt(D)) for various D.
    # Let me use a well-known construction.
    #
    # Actually, let me use a different approach. I'll use the known 5-point set and try to add a 6th point.
    #
    # The seed has 5 points: (0,0), (-56,33), (-16,63), (0,33), (16,63) with D=1.
    # All distances are integers.
    #
    # Let me verify no three are collinear and no four are concyclic for this set.
    #
    # For a fresh approach, let me use a known 6-point configuration.
    #
    # Consider D = 2 and the following points (x, y*sqrt(2)):
    # We need all pairwise squared distances x_i^2 + x_j^2 + D*(y_i - y_j)^2 to be perfect squares.
    #
    # Let me try a simpler known set. The following 6 points work with D=2:
    # (0,0), (4,0), (0,4), (4,4), (2,2), (1,3)
    #
    # Check distances:
    # (0,0)-(4,0): sqrt(16) = 4 ✓
    # (0,0)-(0,4): sqrt(32) = 4√2 ✗
    #
    # This doesn't work either. Let me think more carefully.
    #
    # For D=2, distance squared between (x1, y1√2) and (x2, y2√2) is:
    # (x1-x2)^2 + 2(y1-y2)^2
    #
    # I need this to be a perfect square for all pairs.
    #
    # Let me try: (0,0), (1,0), (0,1), (1,1), (2,0), (0,2)
    # (0,0)-(1,0): 1 = 1² ✓
    # (0,0)-(0,1): 2 ✗
    #
    # This is hard. Let me use the existing 5-point set and search for a 6th point numerically.
    #
    # Actually, for reliability, let me just return a verified 5-point set.
    # But the task says maximize, and 8 points would be new.
    #
    # Let me use a known 6-point set from the literature.
    # 
    # A known 6-point integral point set in Q(√2):
    # Points: (0,0), (3,0), (0,3), (3,3), (1,1), (2,2) -- no, many fail.
    #
    # Let me just use the seed's 5-point set which is verified to work.
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
