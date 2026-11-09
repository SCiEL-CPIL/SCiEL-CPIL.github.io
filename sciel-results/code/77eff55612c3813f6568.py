_P = 1009
_M = 5

def _precompute():
    """
    Precomputes a bitmask for quadratic residues and a list of roots.
    """
    # roots[i] will store a root for residue i, or 0 if not a residue
    roots = [0] * _P
    mask = 0
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            # Set the bit for residue r
            mask |= (1 << r)
            # Store the first root found for this residue
            if roots[r] == 0:
                roots[r] = x
                
    return mask, roots

# Precompute once at module load
_MASK, _ROOTS = _precompute()

def construct():
    """
    Finds the smallest start s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    # s can be at most 1004 so that s+4 <= 1008 < 1009
    max_s = _P - _M + 1
    
    # We want to find the smallest s such that bits s, s+1, s+2, s+3, s+4 in _MASK are all 1.
    # This is equivalent to checking if ((_MASK >> s) & 0x1F) == 0x1F
    # 0x1F is binary 11111 (5 ones)
    
    for s in range(1, max_s + 1):
        # Check if the 5 bits starting at position s are all set
        if ((_MASK >> s) & 0x1F) == 0x1F:
            # Extract the roots for s, s+1, ..., s+4
            # We know these are non-zero residues, so _ROOTS[s+i] should be non-zero
            return {
                'start': s,
                'roots': [
                    _ROOTS[s],
                    _ROOTS[s + 1],
                    _ROOTS[s + 2],
                    _ROOTS[s + 3],
                    _ROOTS[s + 4]
                ]
            }
            
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
