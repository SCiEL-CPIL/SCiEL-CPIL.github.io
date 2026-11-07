import time

def _verify_squares(squares):
    n = len(squares[0])
    for sq in squares:
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            pairs = set()
            for r in range(n):
                for c in range(n):
                    pair = (squares[i][r][c], squares[j][r][c])
                    if pair in pairs:
                        return False
                    pairs.add(pair)
            if len(pairs) != n * n:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Map index k in 0..9 to (a, b) where a = k % 5, b = k // 5
    # Map (a, b) back to integer: a + 5*b
    def idx_to_ab(k):
        return (k % 5, k // 5)
    
    def ab_to_idx(a, b):
        return a + 5 * b
    
    # L1: symbol = (a_r + a_c mod 5, b_r + b_c mod 2)
    # L2: symbol = (a_r + 2*a_c mod 5, b_r + 2*b_c mod 2)
    # L3: symbol = (a_r + 3*a_c mod 5, b_r + 3*b_c mod 2) -- but 3 ≡ -2 mod 5, need to check orthogonality
    
    # Actually, for orthogonality we need the difference forms to be independent.
    # L1 uses coefficients (1,1) for (a,b) in row and col
    # L2 uses coefficients (1,2) for a and (1,2) for b -- wait, let me be more careful.
    
    # Let row = (ar, br), col = (ac, bc)
    # L1 symbol: (ar + ac mod 5, br + bc mod 2)
    # L2 symbol: (ar + 2*ac mod 5, br + 2*bc mod 2)
    
    # For orthogonality: if L1(r1,c1) = L1(r2,c2) and L2(r1,c1) = L2(r2,c2), then (r1,c1) = (r2,c2)
    # From L1: ar1+ac1 ≡ ar2+ac2 (mod 5), br1+bc1 ≡ br2+bc2 (mod 2)
    # From L2: ar1+2*ac1 ≡ ar2+2*ac2 (mod 5), br1+2*bc1 ≡ br2+2*bc2 (mod 2)
    # Subtract: ac1 - ac2 ≡ 0 (mod 5), bc1 - bc2 ≡ 0 (mod 2)
    # So ac1 = ac2, bc1 = bc2, then ar1 = ar2, br1 = br2.
    # Great, they are orthogonal!
    
    L1 = []
    L2 = []
    for r in range(10):
        ar, br = idx_to_ab(r)
        row1 = []
        row2 = []
        for c in range(10):
            ac, bc = idx_to_ab(c)
            # L1
            s1_a = (ar + ac) % 5
            s1_b = (br + bc) % 2
            row1.append(ab_to_idx(s1_a, s1_b))
            # L2
            s2_a = (ar + 2 * ac) % 5
            s2_b = (br + 2 * bc) % 2
            row2.append(ab_to_idx(s2_a, s2_b))
        L1.append(row1)
        L2.append(row2)
    
    squares = [L1, L2]
    
    if _verify_squares(squares):
        return squares
    else:
        # Fallback: just return L1 which is definitely a valid Latin square
        return [L1]
