"""
Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009.
"""
import time

_P = 1009
_M = 5

def construct():
    # Deadline guard (though this task is very fast, it's good practice per directive)
    start_time = time.time()
    
    # 1. Precompute quadratic residues and one root for each
    # is_residue[i] is True if i is a non-zero quadratic residue mod _P
    # root_of[i] stores a witness x such that x*x % _P == i
    is_residue = [False] * _P
    root_of = [0] * _P
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_residue[r] = True
            # Store the first root found (smallest x)
            if root_of[r] == 0:
                root_of[r] = x

    # 2. Search for the minimum s such that s, s+1, ..., s+4 are all non-zero residues
    # s must be between 1 and _P - _M (1004) to ensure s+4 <= 1008
    for s in range(1, _P - _M + 1):
        # Check if all 5 consecutive numbers are residues
        # s, s+1, s+2, s+3, s+4
        valid = True
        for i in range(_M):
            val = s + i
            if not is_residue[val]:
                valid = False
                break
        
        if valid:
            # Construct the result
            roots = [root_of[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
            
        # Safety check for time (should never trigger for P=1009)
        if time.time() - start_time > 80:
            break

    raise RuntimeError("No valid run found")
