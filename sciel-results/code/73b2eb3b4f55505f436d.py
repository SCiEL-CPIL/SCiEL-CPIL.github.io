import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 10
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    # Construct two orthogonal Latin squares of order 10
    # Using linear forms over Z_10
    # L1(i,j) = (i + j) % 10
    # L2(i,j) = (i + 2*j) % 10
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for i in range(n)] for j in range(n)]
    
    # Wait, the list comprehension for L2 above was wrong in my draft.
    # Correct:
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1, L2]
    
    # Verification function
    def is_latin(sq):
        for row in sq:
            if sorted(row) != list(range(n)):
                return False
        for col in zip(*sq):
            if sorted(col) != list(range(n)):
                return False
        return True
        
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n

    # Check validity
    if not is_latin(L1) or not is_latin(L2) or not are_orthogonal(L1, L2):
        # Fallback to single square
        return [L1]
        
    return squares
# EVOLVE-BLOCK-END
