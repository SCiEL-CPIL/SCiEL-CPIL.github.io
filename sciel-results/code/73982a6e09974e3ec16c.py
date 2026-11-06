"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: ensure we finish well within 60s
    start_time = time.time()
    deadline = start_time + 55.0
    
    p = 1009
    
    # Map each quadratic residue to one of its roots
    # We want 1 <= r <= p-1
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
            
    # Search for the minimum s
    # s must be at least 1 and s+4 must be at most p-1 (since 0 is excluded? 
    # Actually "zero residues ... are excluded" usually means the value 0 is not allowed 
    # in the sequence. Since s>=1, s..s+4 are in 1..1008 if s<=1004.
    # If s=1005, s+4=1009=0 mod 1009, which is excluded.
    # So max s is 1004.
    
    for s in range(1, p - 4): # 1 to 1004 inclusive
        if time.time() > deadline:
            # Fallback: return any valid solution if we hit deadline (should not happen)
            # But since the loop is tiny, this is just a safety net.
            # We can just raise an error or return a known valid one if we had one.
            # However, the loop takes microseconds.
            pass
            
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if all((s + i) in roots for i in range(5)):
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # If no solution found (should not happen for p=1009), raise error
    raise RuntimeError("No solution found")
# EVOLVE-BLOCK-END
