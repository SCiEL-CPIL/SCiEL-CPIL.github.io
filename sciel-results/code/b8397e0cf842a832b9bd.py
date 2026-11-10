"""Find the earliest run of 5 consecutive quadratic residues modulo 1009."""
import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute: for each residue value, store a root
    # r^2 mod p = residue
    # We iterate r from 1 to (p-1)//2 to get distinct residues
    root_of = {}
    for r in range(1, (p - 1) // 2 + 1):
        residue = (r * r) % p
        if residue not in root_of:
            root_of[residue] = r
        if time.time() > deadline:
            break
            
    # Build a boolean array: is_residue[x] = True if x is a QR mod p (x in 1..1008)
    is_residue = [False] * p
    for res in root_of:
        is_residue[res] = True
    
    # Search for the minimum s such that s, s+1, s+2, s+3, s+4 are all residues
    # s ranges from 1 to 1004 (since s+4 <= 1008)
    best_result = None
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if (is_residue[s] and is_residue[s + 1] and is_residue[s + 2] 
                and is_residue[s + 3] and is_residue[s + 4]):
            best_result = {
                'start': s,
                'roots': [root_of[s + i] for i in range(5)]
            }
            break  # We found the minimum s, so we can stop
            
    if best_result:
        return best_result
        
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
