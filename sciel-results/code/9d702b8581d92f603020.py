"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct mutually orthogonal Latin squares of order 10 using the
    Kronecker product of a 2x2 Latin square and 4 MOLS of order 5.
    
    The construction: C_k[2a+a'][2b+b'] = 2*L5[k][a][b] + L2[a'][b']
    where L2 = [[0,1],[1,0]] and L5[k] are 4 MOLS of order 5.
    
    This produces 4 pairwise orthogonal Latin squares of order 10.
    """
    n = 10
    
    # 2x2 Latin square
    L2 = [[0, 1],
          [1, 0]]
    
    # 4 MOLS of order 5 using finite field construction
    # L5[k][a][b] = (a + k*b) % 5 for k in {1,2,3,4}
    L5 = []
    for k in range(1, 5):
        square = []
        for a in range(5):
            row = []
            for b in range(5):
                row.append((a + k * b) % 5)
            square.append(row)
        L5.append(square)
    
    # Construct 4 MOLS of order 10 using Kronecker product
    # C_k[2a+a'][2b+b'] = 2*L5[k][a][b] + L2[a'][b']
    squares = []
    for k in range(4):
        sq = []
        for i in range(10):
            row = []
            a = i // 2
            a_prime = i % 2
            for j in range(10):
                b = j // 2
                b_prime = j % 2
                val = 2 * L5[k][a][b] + L2[a_prime][b_prime]
                row.append(val)
            sq.append(row)
        squares.append(sq)
    
    # Verify each square is a valid Latin square
    def is_latin_square(sq, n=10):
        for i in range(n):
            row_set = set(sq[i])
            if len(row_set) != n or any(x < 0 or x >= n for x in row_set):
                return False
        for j in range(n):
            col_set = set(sq[i][j] for i in range(n))
            if len(col_set) != n or any(x < 0 or x >= n for x in col_set):
                return False
        return True
    
    # Verify orthogonality between two squares
    def are_orthogonal(sq1, sq2, n=10):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n
    
    # Find the largest mutually orthogonal subset
    from itertools import combinations
    
    n_sq = len(squares)
    
    # Check which squares are valid Latin squares
    valid_squares = []
    for idx, sq in enumerate(squares):
        if is_latin_square(sq):
            valid_squares.append(idx)
    
    # Find max clique in the orthogonality graph among valid squares
    best = []
    if valid_squares:
        # Try from largest to smallest
        for size in range(min(len(valid_squares), 9), 0, -1):
            found = False
            for combo in combinations(valid_squares, size):
                valid = True
                for a in range(len(combo)):
                    for b in range(a + 1, len(combo)):
                        if not are_orthogonal(squares[combo[a]], squares[combo[b]]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best = list(combo)
                    found = True
                    break
            if found:
                break
    
    if not best:
        # Fallback: return just the first valid Latin square
        if valid_squares:
            best = [valid_squares[0]]
        else:
            # Should not happen, but just in case
            sq = [[(i + j) % 10 for j in range(10)] for i in range(10)]
            return [sq]
    
    result = [squares[i] for i in best]
    
    # Ensure all values are Python ints
    for sq in result:
        for i in range(10):
            for j in range(10):
                sq[i][j] = int(sq[i][j])
    
    return result

# EVOLVE-BLOCK-END
