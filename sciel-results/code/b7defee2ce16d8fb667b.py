# EVOLVE-BLOCK-START
def construct():
    p = 1009
    import time
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    # 1. Identify all quadratic residues
    is_qr = [False] * p
    roots = [0] * p # Store smallest root
    
    for x in range(1, p):
        r = (x * x) % p
        if not is_qr[r]:
            is_qr[r] = True
            roots[r] = x
            
    # 2. Find minimum s
    min_s = -1
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            min_s = s
            break
            
    if min_s == -1:
        # Fallback if not found (should not happen)
        min_s = 1004
        
    res_roots = [roots[min_s + i] for i in range(5)]
    return {'start': min_s, 'roots': res_roots}
# EVOLVE-BLOCK-END
