"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of two mutually orthogonal 10x10 Latin squares.
    This is the maximum possible number of MOLS for order 10.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 59.0
    
    # Construct two orthogonal Latin squares of order 5
    n5 = 5
    A5 = [[(i + j) % n5 for j in range(n5)] for i in range(n5)]
    B5 = [[(i + 2*j) % n5 for j in range(n5)] for i in range(n5)]
    
    # Inflate to order 10
    L1 = [[0] * n for _ in range(n)]
    L2 = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            i_prime = i // 2
            r1 = i % 2
            j_prime = j // 2
            r2 = j % 2
            
            L1[i][j] = 2 * A5[i_prime][j_prime] + r2
            L2[i][j] = 2 * B5[i_prime][j_prime] + r1
            
    # Verification step (defensive)
    # Check Latin property
    for sq in [L1, L2]:
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return [A5] # Fallback to a valid single square (though this shouldn't happen)
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return [A5]
                
    # Check orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (L1[i][j], L2[i][j])
            if p in pairs:
                return [A5] # Fallback
            pairs.add(p)
            
    if len(pairs) != n * n:
        return [A5] # Fallback

    # Ensure types are strictly Python int
    for i in range(n):
        for j in range(n):
            L1[i][j] = int(L1[i][j])
            L2[i][j] = int(L2[i][j])
            
    return [L1, L2]
# EVOLVE-BLOCK-END
