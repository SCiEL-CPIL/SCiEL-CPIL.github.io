# EVOLVE-BLOCK-START

def construct():
    n = 10
    
    # Construct two orthogonal Latin squares using Z5 x Z2 structure
    # Map index k to (k%5, k//5)
    sq1 = [[0]*n for _ in range(n)]
    sq2 = [[0]*n for _ in range(n)]
    
    for i in range(n):
        r1 = i % 5
        r2 = i // 5
        for j in range(n):
            c1 = j % 5
            c2 = j // 5
            
            # Square 1: Group table of Z5 x Z2
            # Symbol = (r1+c1 mod 5) * 2 + (r2+c2 mod 2)
            a1 = (r1 + c1) % 5
            b1 = (r2 + c2) % 2
            sq1[i][j] = int(a1 * 2 + b1)
            
            # Square 2: Orthogonal partner
            # Symbol = (r1+2*c1 mod 5) * 2 + r2
            # Note: b2 = r2 + 2*c2 mod 2 = r2
            a2 = (r1 + 2 * c1) % 5
            b2 = r2
            sq2[i][j] = int(a2 * 2 + b2)
            
    # Verification helpers
    def is_latin(square):
        for i in range(n):
            if len(set(square[i])) != n:
                return False
        for j in range(n):
            col = [square[i][j] for i in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def are_orthogonal(a, b):
        seen = set()
        for i in range(n):
            for j in range(n):
                pair = (a[i][j], b[i][j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    # Check validity
    if is_latin(sq1) and is_latin(sq2) and are_orthogonal(sq1, sq2):
        return [sq1, sq2]
    else:
        # Fallback: return just the first square if something went wrong
        # (This should not happen with the correct algebraic construction)
        return [sq1]

# EVOLVE-BLOCK-END
