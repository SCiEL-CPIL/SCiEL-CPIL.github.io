import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    p = 1009
    # Precompute: for each non-zero residue, store one root
    root_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in root_map:
            root_map[r] = x
    
    # Search ascending to minimize s
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        valid = True
        for i in range(5):
            val = s + i
            if val not in root_map:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [root_map[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
