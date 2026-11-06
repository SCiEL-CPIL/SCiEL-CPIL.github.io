import time
import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _pivot_move(chain, k, axis, angle, rng):
    # Rotate chain[k:] around bond k-1, k
    # Axis is line through chain[k-1] and chain[k]
    # This implementation is complex for general axes.
    # Simplified: Only allow pivots where the bond is aligned with an axis.
    # Then rotate around one of the other two axes.
    pass
