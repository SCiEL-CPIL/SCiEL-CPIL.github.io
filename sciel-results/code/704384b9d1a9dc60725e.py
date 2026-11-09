import time

_P = 1009
_M = 5

def _find_min_run():
    """
    Finds the smallest s such that s, s+1, ..., s+4 are all quadratic residues mod _P.
    Returns a dict with 'start' and 'roots'.
    """
    # Build mapping from residue to smallest root in [1, P-1]
    roots_map = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:  # Exclude zero residue
            if r not in roots_map:
                roots_map[r] = x
    
    # Create a set for O(1) lookup
    residues = set(roots_map.keys())
    
    # Search for the smallest s
    # s must be in [1, P-M] such that s..s+M-1 are all in residues
    # Since P=1009 and M=5, s goes up to 1004.
    # We iterate from 1 upwards to find the minimum.
    
    for s in range(1, _P - _M + 1):
        # Check if s, s+1, ..., s+4 are all quadratic residues
        # Using set membership is very fast for this size
        if (s in residues and 
            (s+1) in residues and 
            (s+2) in residues and 
            (s+3) in residues and 
            (s+4) in residues):
            
            # Extract roots
            roots = [roots_map[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
            
    # Fallback if no run found (should not happen for these parameters)
    return None

# Precompute the result at module load time
# This ensures construct() is O(1) and avoids any runtime overhead
_RESULT = _find_min_run()

def construct():
    """
    Returns the precomputed candidate with the minimum start index.
    """
    if _RESULT is None:
        # This should never happen for p=1009, m=5
        raise RuntimeError("No run of quadratic residues found")
    return _RESULT
