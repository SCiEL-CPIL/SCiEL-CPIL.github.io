import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]
DEADLINE = 130  # safety margin before 150s kill

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _is_valid(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And compare with the standard tensor: T[i,j,k] = 1 if j==k else 0
    # Actually for matrix multiplication, the tensor is T[i,j,k] = delta_{j,k}
    # Wait, no. The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 if j==k, else 0.
    # Let's verify: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = delta_{j,k}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if j == k else 0
                actual = 0
                for r in range(rank):
                    u = U[i * m + j][r]
                    v = V[j * p + k][r]
                    w = W[k * n + i][r]
                    actual += u * v * w
                if actual != expected:
                    return False
    return True

def _try_merge(U, V, W, n, m, p, r1, r2):
    """Try to merge terms r1 and r2 into a single term.
    Returns (new_U, new_V, new_W) if successful, else None.
    Merging is possible if the 3x3 system for the merged term has a solution
    with integer/half-integer entries.
    """
    rank = len(U[0])
    # For each (i,j,k) triple, we need:
    # U1[i*m+j]*V1[j*p+k]*W1[k*n+i] + U2[i*m+j]*V2[j*p+k]*W2[k*n+i] = U_new[i*m+j]*V_new[j*p+k]*W_new[k*n+i]
    # This is generally not solvable by simple merging.
    # A simpler approach: check if term r2 is a scalar multiple of term r1.
    # If U[r1] = c * U[r2], V[r1] = c * V[r2], W[r1] = c * W[r2] for some c,
    # then we can merge by adding them.
    
    # Check if term r2 is a multiple of term r1
    # Find c from the first non-zero entry
    c = None
    for idx in range(len(U[0])):  # This is wrong, U has shape (n*m, rank)
        pass
    
    # Let's check proportionality
    # U[:, r1] and U[:, r2]
    u1 = [U[i][r1] for i in range(n * m)]
    u2 = [U[i][r2] for i in range(n * m)]
    v1 = [V[i][r1] for i in range(m * p)]
    v2 = [V[i][r2] for i in range(m * p)]
    w1 = [W[i][r1] for i in range(p * n)]
    w2 = [W[i][r2] for i in range(p * n)]
    
    # Find c such that u2 = c * u1
    c_val = None
    for a, b in zip(u1, u2):
        if a != 0:
            c_candidate = b / a
            if c_val is None:
                c_val = c_candidate
            elif c_val != c_candidate:
                return None
            break
    
    if c_val is None:
        # Check if u2 is all zeros
        if all(x == 0 for x in u2):
            c_val = 0
        else:
            return None
    
    # Verify v2 = c_val * v1 and w2 = c_val * w1
    for a, b in zip(v1, v2):
        if a * c_val != b:
            return None
    for a, b in zip(w1, w2):
        if a * c_val != b:
            return None
    
    # Merge: new term = term1 + term2
    # We can just remove r2 and add r2's contribution to r1
    new_U = [row[:] for row in U]
    new_V = [row[:] for row in V]
    new_W = [row[:] for row in W]
    
    for i in range(n * m):
        new_U[i][r1] = new_U[i][r1] + new_U[i][r2]
    for i in range(m * p):
        new_V[i][r1] = new_V[i][r1] + new_V[i][r2]
    for i in range(p * n):
        new_W[i][r1] = new_W[i][r1] + new_W[i][r2]
    
    # Remove column r2
    new_U = [row[:r2] + row[r2+1:] for row in new_U]
    new_V = [row[:r2] + row[r2+1:] for row in new_V]
    new_W = [row[:r2] + row[r2+1:] for row in new_W]
    
    return {"U": new_U, "V": new_V, "W": new_W}

def _greedy_merge_reduce(U, V, W, n, m, p, time_limit):
    """Try to reduce rank by greedily merging proportional terms."""
    start_time = time.time()
    current = {"U": U, "V": V, "W": W}
    rank = len(U[0])
    
    while time.time() - start_time < time_limit:
        merged = False
        current_rank = len(current["U"][0])
        for r1 in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            for r2 in range(r1 + 1, current_rank):
                result = _try_merge(current["U"], current["V"], current["W"], n, m, p, r1, r2)
                if result is not None:
                    # Verify
                    if _is_valid(result["U"], result["V"], result["W"], n, m, p):
                        current = result
                        merged = True
                        break
            if merged:
                break
        if not merged:
            break
    
    return current

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try greedy merge reduction with a time budget
    time_budget = 120  # seconds
    elapsed = time.time() - start_time
    if elapsed < time_budget:
        reduced = _greedy_merge_reduce(result["U"], result["V"], result["W"], n, m, p, time_budget - elapsed)
        if len(reduced["U"][0]) < len(result["U"][0]):
            result = reduced
    
    return result
