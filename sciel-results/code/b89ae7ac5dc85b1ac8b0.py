"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    m = 5
    
    # Precompute roots for each quadratic residue
    # We store the smallest root in 1..p-1 for each residue
    root_map = {}
    for x in range(1, p):
        if time.time() > deadline:
            # Fallback: return a known valid late run or raise
            # But this shouldn't happen
            raise TimeoutError("Computation exceeded deadline")
        r = (x * x) % p
        if r not in root_map:
            root_map[r] = x
    
    # Now search for the smallest s such that s, s+1, ..., s+4 are all quadratic residues
    # and none of them are zero (since 0 is excluded by the constraint 1<=ri<=1008)
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            # Return best found so far or raise
            # Actually, we haven't found any yet, so we need to handle this
            raise TimeoutError("Computation exceeded deadline")
        
        all_residues = True
        roots = []
        for i in range(m):
            val = s + i
            if val > p - 1:
                # Wraparound excluded, so if val >= p, it's invalid
                all_residues = False
                break
            if val not in root_map:
                all_residues = False
                break
            roots.append(root_map[val])
        
        if all_residues:
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
