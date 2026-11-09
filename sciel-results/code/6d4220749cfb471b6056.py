import time

def construct():
    p = 1009
    is_residue = [False] * p
    root_map = {}
    
    # Precompute quadratic residues and one root for each
    for x in range(1, p):
        v = (x * x) % p
        if v != 0:  # Exclude zero residue
            is_residue[v] = True
            if v not in root_map:
                root_map[v] = x
    
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all residues
    # s must be at least 1 and s+4 must be at most p-1 (since 0 is excluded)
    # So s ranges from 1 to p-5
    for s in range(1, p - 4):
        if all(is_residue[s + i] for i in range(5)):
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    # This should not happen for p=1009, but just in case
    raise RuntimeError("No solution found")
