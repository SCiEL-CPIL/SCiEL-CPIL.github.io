import math
import time
import itertools
import numpy as np
from typing import Dict, List, Tuple

def construct() -> Dict:
    start_time = time.time()
    deadline = start_time + 140.0

    def is_valid_paley(p):
        # Construct Paley tournament on p vertices
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        # Build in-neighborhood masks
        # Using Python ints is fast enough for p <= 83
        in_mask = [0] * p
        for v in range(p):
            mask = 0
            for u in range(p):
                if u != v and (v - u) % p in squares:
                    mask |= (1 << u)
            in_mask[v] = mask
            
        # Check: for every 4-subset {u,v,w,x}, intersection of in_mask[u], in_mask[v], in_mask[w], in_mask[x] is non-empty
        # We can speed this up by precomputing intersections of pairs?
        # Or just use the nested loop with early exit.
        # For p=67, it's fast.
        
        for u in range(p):
            for v in range(u+1, p):
                inter1 = in_mask[u] & in_mask[v]
                if inter1 == 0:
                    continue
                for w in range(v+1, p):
                    inter2 = inter1 & in_mask[w]
                    if inter2 == 0:
                        continue
                    for x in range(w+1, p):
                        if (inter2 & in_mask[x]) == 0:
                            return False
        return True

    candidates = [31, 43, 47, 59, 67, 71, 83]
    
    for p in candidates:
        if time.time() > deadline - 10.0:
            break
            
        if is_valid_paley(p):
            squares = set()
            for x in range(1, p):
                squares.add((x*x) % p)
            
            orientation = []
            for i in range(p):
                for j in range(i+1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
                        
            return {'n': p, 'orientation': orientation}
            
    # Fallback
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x*x) % p)
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
