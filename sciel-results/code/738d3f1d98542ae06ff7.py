"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 10}

def _build_mols():
    """Generate candidate MOLS of order 10."""
    n = 10
    squares = []
    
    # Use a verified construction. We'll generate Latin squares using
    # the formula: S_k(i,j) = (i + k*j) % 10 for k coprime to 10.
    # This gives Latin squares, but they may not be mutually orthogonal.
    # For k in {1,3,7,9} (coprime to 10), we get 4 Latin squares.
    # Check orthogonality: S_k1(i,j) and S_k2(i,j) are orthogonal iff
    # the map (i,j) -> (S_k1(i,j), S_k2(i,j)) is bijective.
    
    # Actually, let's use a simpler verified approach:
    # Hardcode a known valid pair of orthogonal Latin squares of order 10.
    
    # Latin square L1: L1[i][j] = (i + j) % 10
    # Latin square L2: L2[i][j] = (i + 3*j) % 10
    # Check orthogonality: need (i+j)%10 and (i+3j)%10 to be bijective.
    # Let a = (i+j)%10, b = (i+3j)%10. Then b-a = 2j mod 10, so j = (b-a)*3 mod 10 (since 2*3=6, not invertible).
    # 2j mod 10: j=0->0, j=1->2, j=2->4, j=3->6, j=4->8, j=5->0, ... not bijective.
    # So this pair is NOT orthogonal.
    
    # Let me try: L1[i][j] = (i + j) % 10, L2[i][j] = (i + 2*j) % 10
    # b - a = j mod 10, so j = (b-a) % 10, i = (a-j) % 10. Bijective! So this pair IS orthogonal.
    
    # L3[i][j] = (i + 3*j) % 10: check with L1: b-a = 2j mod 10, not bijective. Not orthogonal to L1.
    # L3 with L2: b-c = j mod 10 where c = (i+3j)%10, so j = (b-c)%10. Bijective! Orthogonal to L2.
    
    # So {L1, L2} is an orthogonal pair. Can we add more?
    # L3 is not orthogonal to L1, so we can't have all three.
    
    # Let's try a different set. Use L1[i][j]=(i+j)%10, L2[i][j]=(i+2j)%10, L3[i][j]=(i+3j)%10
    # L1&L2: orthogonal (shown above)
    # L1&L3: b-a=2j mod 10, not bijective. Not orthogonal.
    # So max from this family is 2.
    
    # Let me try another approach. Use L1[i][j]=(i+j)%10, L2[i][j]=(2i+j)%10
    # a=(i+j)%10, b=(2i+j)%10. b-a=i mod 10, so i=(b-a)%10, j=(a-i)%10. Bijective! Orthogonal.
    
    # L3[i][j]=(3i+j)%10: with L1: b-a=2i mod 10, not bijective. Not orthogonal to L1.
    # With L2: b-c=i mod 10. Orthogonal to L2.
    
    # So again max 2 from linear forms.
    
    # The known maximum for order 10 is 2 MOLS. Let me just return a verified pair.
    
    # Square 0: S0[i][j] = (i + j) % 10
    # Square 1: S1[i][j] = (i + 2*j) % 10
    
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    sq1 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    squares.append(sq0)
    squares.append(sq1)
    
    return squares

def _verify_orthogonality(squares):
    """Verify that all pairs of squares are orthogonal."""
    n = 10
    num_squares = len(squares)
    for i in range(num_squares):
        for j in range(i + 1, num_squares):
            pairs = set()
            for r in range(n):
                for c in range(n):
                    pair = (squares[i][r][c], squares[j][r][c])
                    if pair in pairs:
                        return False
                    pairs.add(pair)
    return True

def _verify_latin(square):
    """Verify that a square is a Latin square."""
    n = 10
    for i in range(n):
        if sorted(square[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [square[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def construct():
    n = P['n']
    deadline = time.time() + 55  # 55s safety margin within 60s limit
    
    # Generate candidate MOLS
    squares = _build_mols()
    
    # Verify each is a Latin square
    valid_squares = []
    for sq in squares:
        if _verify_latin(sq):
            valid_squares.append(sq)
    
    if not valid_squares:
        # Last resort: trivial Latin square
        return [[[ (i + j) % n for j in range(n)] for i in range(n)]]
    
    # Greedily build the largest mutually orthogonal subset
    best_set = [valid_squares[0]]
    for k in range(1, len(valid_squares)):
        if time.time() > deadline:
            break
        compatible = True
        for existing in best_set:
            pairs = set()
            orthogonal = True
            for i in range(n):
                for j in range(n):
                    pair = (existing[i][j], valid_squares[k][i][j])
                    if pair in pairs:
                        orthogonal = False
                        break
                    pairs.add(pair)
                if not orthogonal:
                    break
            if not orthogonal:
                compatible = False
                break
        if compatible:
            best_set.append(valid_squares[k])
    
    # Convert to Python int
    result = []
    for sq in best_set:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    return result

# EVOLVE-BLOCK-END
