"""Rewritten program to find the minimum start of a run of 5 quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 80.0
    
    # Map each quadratic residue to its smallest positive root in [1, p-1]
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
            
    # Search for the minimum start s
    # s ranges from 1 to 1004 (since s+4 <= 1008)
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if all((s + i) in roots for i in range(5)):
            # Extract the roots
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # If no run found (should not happen for p=1009, m=5)
    raise RuntimeError("No run of 5 quadratic residues found")
# EVOLVE-BLOCK-END
