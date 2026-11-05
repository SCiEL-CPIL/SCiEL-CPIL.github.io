"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import itertools
    
    n = 16
    
    # Helper to check validity
    def is_valid(colours):
        # Build matrix
        cm = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                cm[i][j] = colours[idx]
                cm[j][i] = colours[idx]
                idx += 1
        
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            cols = {cm[a][b], cm[a][c], cm[a][d], cm[b][c], cm[b][d], cm[c][d]}
            if len(cols) < 5:
                return False
        return True

    # Candidate 1: Multiplicative mod 17
    # Map vertices 0..15 to labels 1..16
    # Colour = (label_i * label_j) % 17
    # Labels are 1..16, so colours are in 1..16 (0 is not possible as 17 is prime and labels non-zero)
    # We can shift to 0..15
    colours_mult = []
    for i in range(n):
        for j in range(i+1, n):
            li = i + 1
            lj = j + 1
            c = (li * lj) % 17
            # c is in 1..16
            colours_mult.append(c - 1) # 0..15
            
    if is_valid(colours_mult):
        return colours_mult
        
    # Candidate 2: Additive mod 17 (Seed)
    colours_add = []
    for i in range(n):
        for j in range(i+1, n):
            c = (i + j) % 17
            colours_add.append(c)
            
    # This should always be valid
    return colours_add
# EVOLVE-BLOCK-END
