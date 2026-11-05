"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    n = 16
    
    def check_validity(colours, n=16):
        """Check that every K4 has at least 5 distinct colours."""
        colour_matrix = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                colour_matrix[i][j] = colours[idx]
                colour_matrix[j][i] = colours[idx]
                idx += 1
        
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            cols = set()
            cols.add(colour_matrix[a][b])
            cols.add(colour_matrix[a][c])
            cols.add(colour_matrix[a][d])
            cols.add(colour_matrix[b][c])
            cols.add(colour_matrix[b][d])
            cols.add(colour_matrix[c][d])
            if len(cols) < 5:
                return False
        return True
    
    def count_colours(colours):
        return len(set(colours))
    
    # Try various non-linear constructions based on vertex labels
    
    best_colours = None
    best_count = 120  # worst case
    
    # Construction 1: XOR-based
    # Assign each vertex a 4-bit label, edge colour = XOR of labels
    # This gives colours in 0..15, so at most 16 colours
    labels_xor = list(range(16))  # identity mapping
    colours_xor = [labels_xor[i] ^ labels_xor[j] for i in range(16) for j in range(i+1, 16)]
    if check_validity(colours_xor):
        c = count_colours(colours_xor)
        if c < best_count:
            best_count = c
            best_colours = colours_xor
    
    # Construction 2: Multiplication mod prime
    # Try primes >= 17
    for p in [17, 19, 23, 29, 31]:
        # Assign vertex i label (i+1) mod p (avoid 0)
        labels = [(i + 1) % p for i in range(16)]
        colours = [(labels[i] * labels[j]) % p for i in range(16) for j in range(i+1, 16)]
        if check_validity(colours):
            c = count_colours(colours)
            if c < best_count:
                best_count = c
                best_colours = colours
    
    # Construction 3: Sum times a factor mod m
    for m in range(13, 20):
        for a in range(1, m):
            colours = [(i + a * j) % m for i in range(16) for j in range(i+1, 16)]
            if check_validity(colours):
                c = count_colours(colours)
                if c < best_count:
                    best_count = c
                    best_colours = colours
    
    # Construction 4: Difference-based (absolute difference)
    colours_diff = [abs(i - j) for i in range(16) for j in range(i+1, 16)]
    if check_validity(colours_diff):
        c = count_colours(colours_diff)
        if c < best_count:
            best_count = c
            best_colours = colours_diff
    
    # Construction 5: Product mod m for various m
    for m in range(13, 30):
        colours = [(i * j) % m for i in range(16) for j in range(i+1, 16)]
        if check_validity(colours):
            c = count_colours(colours)
            if c < best_count:
                best_count = c
                best_colours = colours
    
    # Construction 6: Try (i*j + i + j) mod m
    for m in range(13, 30):
        colours = [(i * j + i + j) % m for i in range(16) for j in range(i+1, 16)]
        if check_validity(colours):
            c = count_colours(colours)
            if c < best_count:
                best_count = c
                best_colours = colours
    
    # Construction 7: Try (i*j - i - j) mod m
    for m in range(13, 30):
        colours = [(i * j - i - j) % m for i in range(16) for j in range(i+1, 16)]
        if check_validity(colours):
            c = count_colours(colours)
            if c < best_count:
                best_count = c
                best_colours = colours
    
    # Construction 8: Bitwise operations
    # Edge colour = (i & j) | (i ^ j) = i | j
    colours_or = [i | j for i in range(16) for j in range(i+1, 16)]
    if check_validity(colours_or):
        c = count_colours(colours_or)
        if c < best_count:
            best_count = c
            best_colours = colours_or
    
    # Edge colour = i & j
    colours_and = [i & j for i in range(16) for j in range(i+1, 16)]
    if check_validity(colours_and):
        c = count_colours(colours_and)
        if c < best_count:
            best_count = c
            best_colours = colours_and
    
    # Construction 9: Try random label assignments with XOR
    random.seed(42)
    for trial in range(100):
        labels = random.sample(range(256), 16)
        colours = [labels[i] ^ labels[j] for i in range(16) for j in range(i+1, 16)]
        if check_validity(colours):
            c = count_colours(colours)
            if c < best_count:
                best_count = c
                best_colours = colours
    
    # Construction 10: Try (i+j) mod m for small m (the original approach)
    for m in range(13, 18):
        colours = [(i + j) % m for i in range(16) for j in range(i+1, 16)]
        if check_validity(colours):
            c = count_colours(colours)
            if c < best_count:
                best_count = c
                best_colours = colours
    
    # If nothing found, fall back to (i+j) % 17
    if best_colours is None:
        best_colours = [(i + j) % 17 for i in range(16) for j in range(i+1, 16)]
    
    return best_colours
# EVOLVE-BLOCK-END
