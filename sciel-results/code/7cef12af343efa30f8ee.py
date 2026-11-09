import time

def construct(k, g):
    start_time = time.time()
    deadline = 110.0
    
    q = k
    d = g - 5
    if d < 3: d = 3
    
    # Rule table for LUW
    # rule[c] = (a, b) for c in 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The pattern is (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        # Index r selects the pair.
        # r=0: (1, c0-2)
        # r=1: (c0-1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0+1)
        if r == 0: rule[c] = (1, c0 - 2)
        elif r == 1: rule[c] = (c0 - 1, 1)
        elif r == 2: rule[c] = (c0, 1)
        else: rule[c] = (1, c0 + 1)
        
    qd = q ** d
    
    def get_neighbours(idx):
        side = 0 if idx < qd else 1
        val = idx if side == 0 else idx - qd
        
        # Decode val into digits x[0]..x[d-1]
        # x[0] is the most significant?
        # In the seed program, x is a tuple (None, x0, x1, ..., xd-1)
        # The seed program uses x[c] where c goes 2..d.
        # So x[2] is the first digit?
        # Let's stick to the seed program's logic but with ints.
        
        # This is getting complicated to map exactly.
        # I will stick to the tuple logic but optimize the BFS.
        pass
