"""Search for a rank-10 tensor decomposition for 2x2 by 2x3 matrix multiplication."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START = time.time()
_HARD_LIMIT = 145.0

def _elapsed():
    return time.time() - _START

def _remaining():
    return max(0.0, _HARD_LIMIT - _elapsed())

def _known_rank11():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen-style 2x2 block (columns 0 and 1 of B)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1

    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1

    # Schoolbook for column 2 of B
    U[0][7] = 1; V[2][7] = 1
    U[1][8] = 1; V[5][8] = 1
    U[2][9] = 1; V[2][9] = 1
    U[3][10] = 1; V[5][10] = 1

    W[4][7] = 1; W[4][8] = 1
    W[5][9] = 1; W[5][10] = 1

    return U, V, W

def _tensor_constraints(n, m, p):
    """Return list of (a, b, c) indices where T[a,b,c] = 1."""
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                constraints.append((a, b, c))
    return constraints

def _verify_tensor(U, V, W, constraints):
    """Check tensor equation exactly: sum_r U[a,r]*V[b,r]*W[c,r] == 1 for all (a,b,c)."""
    rank = len(U[0])
    for a, b, c in constraints:
        total = 0
        for r in range(rank):
            total += U[a][r] * V[b][r] * W[c][r]
        if total != 1:
            return False
    return True

def _construct():
    n, m, p = DIMENSIONS
    U0, V0, W0 = _known_rank11()
    best = (11, U0, V0, W0)
    constraints = _tensor_constraints(n, m, p)

    if _remaining() < 3.0:
        return {"U": best[1], "V": best[2], "W": best[3]}

    # Systematic search: try all pairs (r1, r2) with various coefficient pairs (a, b)
    # where a != b. The merged column is a*col[r1] + b*col[r2].
    # For this to work, the tensor equation must still hold.
    
    coefficients = [(-3, -2), (-3, -1), (-3, 1), (-3, 2), (-3, 3),
                    (-2, -1), (-2, 1), (-2, 2), (-2, 3),
                    (-1, 1), (-1, 2), (-1, 3),
                    (1, 2), (1, 3), (2, 3)]
    
    # Also try (1, -1), (2, -1), etc.
    coefficients += [(1, -1), (2, -1), (3, -1), (1, -2), (2, -2), (3, -2), (1, -3), (2, -3), (3, -3)]
    # And (a, -a) cases
    coefficients += [(1, -1), (2, -2), (3, -3)]
    
    # Deduplicate
    seen = set()
    unique_coeffs = []
    for c in coefficients:
        if c not in seen:
            seen.add(c)
            unique_coeffs.append(c)
    
    trials = 0
    max_trials = 200000
    
    # Phase 1: Systematic search over all pairs and coefficient pairs
    for r1 in range(11):
        for r2 in range(r1 + 1, 11):
            if _remaining() < 5.0:
                break
            for a, b in unique_coeffs:
                trials += 1
                if trials > max_trials or _remaining() < 5.0:
                    break
                
                # Build rank-10 matrices
                rank = 10
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                # Map old columns to new columns
                col_map = {}
                new_idx = 0
                for r in range(11):
                    if r == r1 or r == r2:
                        continue
                    col_map[r] = new_idx
                    new_idx += 1
                combined_idx = new_idx  # should be 9
                
                # Fill non-combined columns
                for r in range(11):
                    if r == r1 or r == r2:
                        continue
                    c = col_map[r]
                    for row in range(n * m):
                        U[row][c] = U0[row][r]
                    for row in range(m * p):
                        V[row][c] = V0[row][r]
                    for row in range(p * n):
                        W[row][c] = W0[row][r]
                
                # Fill the combined column
                for row in range(n * m):
                    U[row][combined_idx] = a * U0[row][r1] + b * U0[row][r2]
                for row in range(m * p):
                    V[row][combined_idx] = a * V0[row][r1] + b * V0[row][r2]
                for row in range(p * n):
                    W[row][combined_idx] = a * W0[row][r1] + b * W0[row][r2]
                
                # Check tensor equation
                if _verify_tensor(U, V, W, constraints):
                    best = (10, U, V, W)
                    return {"U": U, "V": V, "W": W}
    
    # Phase 2: Random search with wider coefficient range if systematic didn't find
    if _remaining() > 5.0:
        rng = random.Random(20250101)
        for _ in range(100000):
            if _remaining() < 5.0:
                break
            r1 = rng.randrange(11)
            r2 = rng.randrange(11)
            if r1 == r2:
                continue
            a = rng.randint(-5, 5)
            b = rng.randint(-5, 5)
            if a == b:
                continue
            
            rank = 10
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col_map = {}
            new_idx = 0
            for r in range(11):
                if r == r1 or r == r2:
                    continue
                col_map[r] = new_idx
                new_idx += 1
            combined_idx = new_idx
            
            for r in range(11):
                if r == r1 or r == r2:
                    continue
                c = col_map[r]
                for row in range(n * m):
                    U[row][c] = U0[row][r]
                for row in range(m * p):
                    V[row][c] = V0[row][r]
                for row in range(p * n):
                    W[row][c] = W0[row][r]
            
            for row in range(n * m):
                U[row][combined_idx] = a * U0[row][r1] + b * U0[row][r2]
            for row in range(m * p):
                V[row][combined_idx] = a * V0[row][r1] + b * V0[row][r2]
            for row in range(p * n):
                W[row][combined_idx] = a * W0[row][r1] + b * W0[row][r2]
            
            if _verify_tensor(U, V, W, constraints):
                best = (10, U, V, W)
                return {"U": U, "V": V, "W": W}
    
    return {"U": best[1], "V": best[2], "W": best[3]}

def construct():
    try:
        return _construct()
    except Exception:
        U, V, W = _known_rank11()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
