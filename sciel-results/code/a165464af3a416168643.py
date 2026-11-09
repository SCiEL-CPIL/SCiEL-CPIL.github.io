import time
import random
from itertools import combinations

_CACHE = {}

def is_not_2_colorable(n, edges):
    key = (n, frozenset(edges))
    if key in _CACHE:
        return _CACHE[key]
    
    # ... implementation ...
    
    result = not backtrack(0)
    _CACHE[key] = result
    return result
