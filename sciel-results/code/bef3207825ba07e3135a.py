import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin

    # Precompute quadratic residues and one root for each
    is_residue = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if not is_residue[r]:
            is_residue[r] = True
            roots[r] = x

    # Search for the minimum start s
    # s must be such that s, s+1, s+2, s+3, s+4 are all residues
    # s ranges from 1 to 1004 (so s+4 <= 1008)
    for s in range(1, p - 4):
        if time.time() > deadline:
            break
        if (is_residue[s] and is_residue[s+1] and is_residue[s+2] and 
            is_residue[s+3] and is_residue[s+4]):
            return {
                'start': s,
                'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            }
            
    # This should never happen for p=1009
    # If we hit the deadline or loop end, raise or return best effort?
    # The task says "Minimize s". If we didn't find one, it's an error.
    raise RuntimeError("No run of 5 quadratic residues found or timeout")
# EVOLVE-BLOCK-END
