"""
Program to find a tournament on n vertices where every set of 4 vertices has a common dominator.
Returns the known Paley tournament on n=67.
"""

import time

def construct():
    # Known best solution: Paley tournament on p=67
    p_known = 67
    squares = {x * x % p_known for x in range(1, p_known)}
    orientation_known = []
    for i in range(p_known):
        for j in range(i + 1, p_known):
            if (j - i) % p_known in squares:
                orientation_known.append(0)
            else:
                orientation_known.append(1)
    
    return {'n': p_known, 'orientation': orientation_known}
