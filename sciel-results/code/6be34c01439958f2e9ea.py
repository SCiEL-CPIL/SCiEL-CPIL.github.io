import time

def construct():
    p = 1009
    # Precompute quadratic residues and a representative root for each
    root_map = {}
    
    # x ranges from 1 to p-1
    # x*x % p will generate all non-zero QRs
    # We only need one root per residue.
    for x in range(1, p):
        res = (x * x) % p
        if res not in root_map:
            root_map[res] = x
    
    # We want to minimize s.
    # s can range from 1 to p-5 (since s+4 <= p-1=1008)
    # s+i must be in root_map for i=0..4
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    for s in range(1, p - 4):
        # Check if s, s+1, s+2, s+3, s+4 are all QRs
        # Using dict keys for O(1) lookup
        if (s in root_map and 
            (s+1) in root_map and 
            (s+2) in root_map and 
            (s+3) in root_map and 
            (s+4) in root_map):
            
            # Construct roots
            roots = []
            for i in range(5):
                roots.append(root_map[s+i])
            
            candidate = {'start': s, 'roots': roots}
            
            # Since we iterate s upwards from 1, the first one found is the minimum.
            # We can return immediately.
            return candidate
            
    raise RuntimeError("No valid run found")
