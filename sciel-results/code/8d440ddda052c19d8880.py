"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from math import gcd

def construct():
    # We return a 5-point set with integer coordinates, integer pairwise distances,
    # no three collinear, no four concyclic.
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
