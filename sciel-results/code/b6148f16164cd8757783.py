import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 60  # 60s safety margin before 90s kill
    
    p = 1009
    # Precompute: for each non-zero residue, store one root
    # Use a list for faster lookup than dict
    root_arr = [0] * p  # root_arr[r] = x if r is a non-zero QR, else 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_arr[r] == 0:
            root_arr[r] = x
    
    # Search ascending to minimize s
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        valid = True
        for i in range(5):
            val = s + i
            if root_arr[val] == 0:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [root_arr[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
