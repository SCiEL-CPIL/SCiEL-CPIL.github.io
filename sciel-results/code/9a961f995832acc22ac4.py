import sys
import random
from collections import deque

def construct(k, g):
    # Known small cages
    known_cages = {
        (3, 5): (10, [(0,1),(0,4),(0,5),(1,2),(1,5),(1,9),(2,3),(2,8),(2,9),(3,4),(3,7),(3,8),(4,6),(4,7),(5,6),(5,7),(6,8),(6,9),(7,8),(7,9)]),
        (3, 6): (14, [(0,1),(0,5),(0,10),(1,2),(1,6),(1,11),(2,3),(2,7),(2,12),(3,4),(3,8),(3,13),(4,5),(4,9),(4,13),(5,6),(5,10),(5,11),(6,7),(6,11),(6,12),(7,8),(7,12),(7,13),(8,9),(8,13),(8,14),(9,10),(9,14),(10,11),(10,14),(11,12),(12,13),(13,14)]),
    }
    
    if (k, g) in known_cages:
        return known_cages[(k, g)]
        
    if k == 3 and g == 16:
        return construct_3_16()
        
    # Fallback to LUW for other cases
    # ... (keep existing LUW code) ...
