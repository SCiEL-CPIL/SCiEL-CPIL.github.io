def construct():
    n = 29
    # Calculate quadratic residues modulo 29
    # x^2 % 29 for x in 1..14
    squares = set()
    for x in range(1, n // 2 + 1):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            # Color 0 if difference is a quadratic residue, else 1
            # This creates a self-complementary graph structure
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
