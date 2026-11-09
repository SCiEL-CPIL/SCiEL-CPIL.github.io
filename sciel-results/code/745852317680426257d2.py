"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a tournament on n vertices satisfying the 4-dominator property.
    Uses the Paley tournament of order 67, which is known to satisfy the condition.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Paley tournament on 67 vertices
    p = 67
    
    # Compute quadratic residues modulo p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Construct orientation bits
    # One bit per pair (i, j) with i < j
    # 0 means i -> j
    # 1 means j -> i
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    
    # Optional: Verify validity if time permits
    # For n=67, binom(67,4) ~ 730k. Verification is fast.
    if time.time() < deadline - 10:
        valid = True
        n = p
        # Build adjacency matrix for quick lookup
        # adj[i][j] = 1 if i -> j else 0
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                if bit == 0:
                    adj[i][j] = 1
                    adj[j][i] = 0
                else:
                    adj[i][j] = 0
                    adj[j][i] = 1
                idx += 1
        
        # Check all 4-sets
        # A 4-set {a,b,c,d} is valid if there exists a vertex v in the set
        # such that v -> u for all u in set \ {v}
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check a
                        if adj[a][b] and adj[a][c] and adj[a][d]:
                            continue
                        # Check b
                        if adj[b][a] and adj[b][c] and adj[b][d]:
                            continue
                        # Check c
                        if adj[c][a] and adj[c][b] and adj[c][d]:
                            continue
                        # Check d
                        if adj[d][a] and adj[d][b] and adj[d][c]:
                            continue
                        # No dominator found
                        valid = False
                        break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to known valid Paley 83 if 67 somehow fails
            # (This should not happen based on problem statement)
            p = 83
            squares = set()
            for x in range(1, p):
                squares.add((x * x) % p)
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (j - i) % p
                    if diff in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
    
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
