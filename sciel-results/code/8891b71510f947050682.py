# ... (top of file)
import itertools
import numpy as np
import math

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    # Try cyclic groups first (fast)
    # ... (existing cyclic search, maybe limited range) ...
    
    # Try Z_2 x Z_2 x Z_5 x Z_16
    # ...
    
    return _construct_fallback(4, g)
