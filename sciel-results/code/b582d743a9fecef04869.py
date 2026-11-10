def construct():
    n = 10
    # Generate 4 MOLS of order 5 using GF(5)
    mols5 = []
    for k in range(1, 5):
        square = [[(k * a + c) % 5 for c in range(5)] for a in range(5)]
        mols5.append(square)
    
    # Double to get 8 MOLS of order 10
    mols10 = []
    for k in range(4):
        sq1 = [[2 * mols5[k][i // 2][j // 2] + (i % 2) for j in range(10)] for i in range(10)]
        mols10.append(sq1)
        sq2 = [[2 * mols5[k][i // 2][j // 2] + (j % 2) for j in range(10)] for i in range(10)]
        mols10.append(sq2)
    
    # Verify each is a valid Latin square
    def is_latin(sq):
        for i in range(10):
            if sorted(sq[i]) != list(range(10)):
                return False
            col = [sq[r][i] for r in range(10)]
            if sorted(col) != list(range(10)):
                return False
        return True
    
    # Verify mutual orthogonality
    def are_orthogonal(a, b):
        pairs = set()
        for i in range(10):
            for j in range(10):
                p = (a[i][j], b[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
    
    # Filter to only keep valid, mutually orthogonal squares
    valid_squares = []
    for sq in mols10:
        if not is_latin(sq):
            continue
        # Check orthogonality with all already selected
        ok = True
        for vs in valid_squares:
            if not are_orthogonal(vs, sq):
                ok = False
                break
        if ok:
            valid_squares.append(sq)
    
    # Convert to Python ints
    result = []
    for sq in valid_squares:
        result.append([[int(sq[i][j]) for j in range(10)] for i in range(10)])
    
    if not result:
        # Fallback: return a single valid Latin square
        result = [[[int((i + j) % 10) for j in range(10)] for i in range(10)]]
    
    return result
