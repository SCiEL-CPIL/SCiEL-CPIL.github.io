<<<<<<< SEARCH
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k != 3:
        # Fallback for other k
        # Use LUW with q=k if k is prime power, else random
        # For this task, assume k=3
        pass

    q = 3
    
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1
=======
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k != 3:
        return 0, []

    q = 3
    
    def girth_of_d(d):
        # LUW(3,d) has girth at least d+5 (for odd d) or d+6 (for even d)
        # Actually the girth is 2*ceil(d/2)+4 or similar; use conservative bound
        return d + 5

    # Find smallest d such that girth >= g
    d = 3
    while girth_of_d(d) < g:
        d += 1
>>>>>>> REPLACE

