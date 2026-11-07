"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 10}

def construct():
    n = P['n']
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    # L1: (i + j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second square L2 orthogonal to L1.
    # L2 is of the form (i + a*j) % n.
    # a must be coprime to n for L2 to be a Latin square.
    # For n=10, possible a values (excluding 1 which is L1) are 3, 7, 9.
    for a in [3, 7, 9]:
        if time.time() > deadline:
            break
        L2 = [[(i + a*j) % n for j in range(n)] for i in range(n)]
        
        # Verify orthogonality
        pairs = set()
        orthogonal = True
        for i in range(n):
            for j in range(n):
                pair = (L1[i][j], L2[i][j])
                if pair in pairs:
                    orthogonal = False
                    break
                pairs.add(pair)
            if not orthogonal:
                break
        
        if orthogonal:
            return [L1, L2]
            
    # If no orthogonal partner found (which is the case for n=10), return just L1
    return [L1]

# EVOLVE-BLOCK-END
