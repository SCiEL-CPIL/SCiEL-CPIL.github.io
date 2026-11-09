import random
from math import gcd

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            gcd_mat[i][j] = gcd(i, j)

    # Precompute conflict masks
    # conflict[x][y] is a bitmask of z such that gcd(x,y)==gcd(y,z)==gcd(z,x)
    # Note: z can be x or y? 
    # The triple must be distinct.
    # If z==x, gcd(x,y)==gcd(y,x)==gcd(x,x) => gcd(x,y)==x.
    # If z==y, gcd(x,y)==gcd(y,y)==gcd(y,x) => gcd(x,y)==y.
    # The problem says "three distinct elements".
    # So when checking if adding x creates a forbidden triple with y, z in S,
    # we know x, y, z are distinct because x is not in S, and y, z are in S (distinct).
    # So we just need to know which z form a forbidden triple with x and y.
    
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            g_xy = gcd_mat[x][y]
            mask = 0
            for z in range(1, N + 1):
                if z == x or z == y:
                    continue
                if gcd_mat[y][z] == g_xy and gcd_mat[z][x] == g_xy:
                    mask |= (1 << (z - 1))
            conflict[x][y] = mask
            # Symmetric for (y, x) if we want, but we will check both or just one?
            # can_add(x, S) checks pairs (y, z) in S.
            # It checks if (x, y, z) is forbidden.
            # This is symmetric in y and z?
            # gcd(x,y)==gcd(y,z)==gcd(z,x).
            # Yes, symmetric.
            # So conflict[x][y] should contain z.
            # When adding x, we iterate y in S. We check if S contains any z in conflict[x][y].
            # This works.

    def can_add_mask(x, S_mask):
        # Check if x can be added to set represented by S_mask
        # Iterate over y in S
        # We can iterate 1..N and check bit
        for y in range(1, N + 1):
            if S_mask & (1 << (y - 1)):
                if S_mask & conflict[x][y]:
                    return False
        return True

    def greedy_solve(seed_val):
        random.seed(seed_val)
        S_mask = 0
        candidates = numbers.copy()
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add_mask(x, S_mask):
                S_mask |= (1 << (x - 1))
        return S_mask

    def simulated_annealing(initial_mask, max_iters=50000):
        S_mask = initial_mask
        best_mask = initial_mask
        best_size = bin(initial_mask).count("1")
        
        # Precompute list of set bits for faster iteration?
        # Or just use the range loop which is fast enough (128 iterations).
        
        for it in range(max_iters):
            # 1. Try to add
            if random.random() < 0.5:
                # Pick random x not in S
                # To pick uniformly from complement, we can pick random x and check
                # Or maintain a list of available?
                # Given N=128, picking random and checking is fine.
                x = random.randint(1, N)
                if not (S_mask & (1 << (x - 1))):
                    if can_add_mask(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        size = bin(S_mask).count("1")
                        if size > best_size:
                            best_size = size
                            best_mask = S_mask
            else:
                # 2. Try swap
                # Pick x in S
                # To pick random x in S, we can use random.choice on a list of set bits
                # But creating list every time is O(N).
                # We can maintain a list S_list synchronized with S_mask.
                # Let's do that for efficiency in swap selection.
                
                # Actually, let's just pick random x, if in S, proceed.
                # Probability of picking x in S is |S|/N ~ 0.7.
                # So we retry often.
                # Better:
                # Generate a random number r in [0, |S|-1] and find the r-th set bit.
                # This is O(N) to find.
                
                # Let's stick to simple: pick random x. If x in S, try swap.
                x = random.randint(1, N)
                if S_mask & (1 << (x - 1)):
                    # Pick y not in S
                    y = random.randint(1, N)
                    if not (S_mask & (1 << (y - 1))):
                        # Check if S \ {x} U {y} is valid
                        # This is can_add(y, S_mask without x)
                        S_temp = S_mask & ~(1 << (x - 1))
                        if can_add_mask(y, S_temp):
                            S_mask = S_temp | (1 << (y - 1))
                            # Size unchanged, so no update to best_size unless we track best_mask for same size?
                            # We only care about max size.
                            
        return best_mask

    best_mask = 0
    best_size = 0
    
    for i in range(20):
        init_mask = greedy_solve(i)
        sa_mask = simulated_annealing(init_mask, max_iters=20000)
        size = bin(sa_mask).count("1")
        if size > best_size:
            best_size = size
            best_mask = sa_mask
            
    # Convert mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    return result
