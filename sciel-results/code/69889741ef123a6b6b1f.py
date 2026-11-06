"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0
    
    # Pre-compute roots for all quadratic residues
    # Using setdefault for slight efficiency over 'if not in'
    roots = {}
    for x in range(1, 1009):
        r = (x * x) % 1009
        roots.setdefault(r, x)
            
    # Ascending search finds the minimum s
    # Since we want to minimize s, we start from 1
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all in roots
        # This implies they are non-zero quadratic residues
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
            
    raise RuntimeError('No valid run found within deadline')
# EVOLVE-BLOCK-END
