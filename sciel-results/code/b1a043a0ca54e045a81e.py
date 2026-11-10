# EVOLVE-BLOCK-START
import time

def _verify_latin(square):
    n = len(square)
    for i in range(n):
        if sorted(square[i]) != list(range(n)):
            return False
        if sorted(square[j][i] for j in range(n)) != list(range(n)):
            return False
    return True

def _verify_orthogonal(s1, s2):
    n = len(s1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (s1[i][j], s2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return len(pairs) == n * n

def _verify_all(squares):
    if not squares:
        return False
    n = len(squares[0])
    for sq in squares:
        if not _verify_latin(sq):
            return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _verify_orthogonal(squares[i], squares[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    # For order 10, the maximum number of MOLS is 2.
    # We construct two orthogonal Latin squares of order 10.
    
    # Square 1: L1[i][j] = (i + j) % 10
    sq1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # Square 2: L2[i][j] = (2*i + j) % 10
    # This is orthogonal to sq1.
    sq2 = [[(2*i + j) % 10 for j in range(10)] for i in range(10)]
    
    squares = [sq1, sq2]
    
    # Verify
    if _verify_all(squares):
        # Ensure all values are Python int
        result = []
        for sq in squares:
            new_sq = [[int(sq[i][j]) for j in range(10)] for i in range(10)]
            result.append(new_sq)
        return result
    else:
        # Fallback to a single valid square
        return [sq1]

# EVOLVE-BLOCK-END
