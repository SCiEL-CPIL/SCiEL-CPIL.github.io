import numpy as np
import itertools
import time

D = 11

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Generate all candidate vectors with exactly 4 non-zero entries of +/- 1
    # There are C(11,4) * 2^4 = 330 * 16 = 5280 candidates
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    
    # We need to select 165 vectors such that for any u, v in selection:
    # u . v <= 2
    # Note: u . u = 4.
    # If u . v = 3, dist^2 = 2, dist = sqrt(2) < 2 (max norm). Invalid.
    # If u . v = 2, dist^2 = 4, dist = 2. Valid.
    
    selected = []
    for v in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 165:
            break
        
        # Check compatibility with all previously selected vectors
        compatible = True
        for u in selected:
            dot = np.dot(u, v)
            if dot > 2:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
    
    # If we didn't find 165, we might need to be smarter, 
    # but greedy on a well-ordered list often works for these lattices.
    # The order of candidates matters.
    
    return np.array(selected, dtype=np.int64)
