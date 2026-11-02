"""Tensor decomposition for 2x2 by 2x4 matrix multiplication, rank 14."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import random
import itertools

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                if a == a_prime and b == b_prime:
                    continue
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val) > 1e-9:
                    return False
    return True

def _try_local_search(U, V, W, deadline, max_iter=5000):
    """Try to reduce rank by finding redundant terms and recombining."""
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    # Check for linearly dependent columns in U, V, W
    # If column r1 and r2 are linearly dependent across all three matrices,
    # we might be able to eliminate one term
    
    # For each pair of columns, check if one can be expressed as a linear combination
    # of the other (i.e., U[:,r1] = c*U[:,r2], V[:,r1] = c*V[:,r2], W[:,r1] = c*W[:,r2])
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            # Check if U[:,r1] is proportional to U[:,r2]
            # Find a nonzero entry in U[:,r2]
            c_val = None
            for i in range(n * m):
                if U[i][r2] != 0:
                    if U[i][r1] == 0:
                        break
                    c_val = U[i][r1] / U[i][r2]
                    break
            else:
                continue
            
            if c_val is None:
                continue
            
            # Verify proportionality for U
            valid = True
            for i in range(n * m):
                if abs(U[i][r1] - c_val * U[i][r2]) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Verify proportionality for V
            for j in range(m * p):
                if abs(V[j][r1] - c_val * V[j][r2]) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Verify proportionality for W
            for k in range(p * n):
                if abs(W[k][r1] - c_val * W[k][r2]) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Found a redundant column! Remove it
            new_U = [row[:r1] + row[r1+1:] for row in U]
            new_V = [row[:r1] + row[r1+1:] for row in V]
            new_W = [row[:r1] + row[r1+1:] for row in W]
            
            if _verify_decomposition(new_U, new_V, new_W):
                return {"U": new_U, "V": new_V, "W": new_W}
    
    return None

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Start with the known rank-14 solution
    best_result = _build_rank14()
    
    # Verify the base solution
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"]):
        # Should not happen, but return anyway
        return best_result
    
    # Try local search to reduce rank
    if time.time() < deadline:
        reduced = _try_local_search(best_result["U"], best_result["V"], best_result["W"], deadline)
        if reduced is not None:
            best_result = reduced
    
    # If we still have rank 14, try a few random perturbations and see if we can
    # find a valid rank-13 or rank-12 decomposition
    if time.time() < deadline - 10:
        current_rank = len(best_result["U"][0])
        if current_rank > 12:
            # Try random initialization with rank 13 and 12
            for target_rank in [13, 12]:
                if time.time() >= deadline - 5:
                    break
                for attempt in range(100):
                    if time.time() >= deadline - 5:
                        break
                    # Random initialization with small integers
                    U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(4)]
                    V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(8)]
                    W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(8)]
                    
                    if _verify_decomposition(U, V, W):
                        best_result = {"U": U, "V": V, "W": W}
                        break
    
    return best_result
# EVOLVE-BLOCK-END
