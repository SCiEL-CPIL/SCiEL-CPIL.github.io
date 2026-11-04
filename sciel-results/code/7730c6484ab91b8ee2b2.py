"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 7]

def _is_valid(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Verify integer/half-integer entries
    for c in range(rank):
        for i in range(n*m):
            val = U[i][c]
            if val != val:  # NaN check
                return False
            if abs(val*2 - round(val*2)) > tol:
                return False
        for j in range(m*p):
            val = V[j][c]
            if val != val:
                return False
            if abs(val*2 - round(val*2)) > tol:
                return False
        for k in range(p*n):
            val = W[k][c]
            if val != val:
                return False
            if abs(val*2 - round(val*2)) > tol:
                return False
    # Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor T for matrix multiplication has T[i,j,k] = 1 for all i,j,k
                # because C[i,k] = sum_j A[i,j]*B[j,k], and the tensor coefficient for
                # the monomial A_ij * B_jk in C_ik is 1 for each (i,j,k)
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Best known valid decomposition (schoolbook, rank 42)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    assert _is_valid(best_U, best_V, best_W)
    
    # Try to find lower-rank decompositions using random search + local improvement
    # Strategy: start with a random rank-r decomposition, try to make it exact
    
    def rand_half_int():
        """Random value from {-2,-1.5,-1,-0.5,0,0.5,1,1.5,2}"""
        return random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    
    def rand_binary():
        return random.choice([0, 1])
    
    # Known lower bounds: for 2x3x7, the rank is at least max(n*m, m*p, n*p) = max(6,21,14) = 21
    # Actually Strassen's bound and others... Let's try ranks from 21 upward
    
    # The number of tensor entries is n*m*p = 42. We need sum over c of U_ic V_jc W_kc = 1 for each (i,j,k).
    # A rank-r decomposition gives r constraints per (i,j,k) entry... no, it gives exactly r terms summing to 1.
    
    # Let's try a structured approach: for 2x3x7, consider that we need
    # For each of the 2*7=14 output entries C[i,k], we have 3 terms: sum_j A[i,j]*B[j,k]
    # So the tensor has 42 ones total (one per (i,j,k)).
    
    # Try rank 21 through 41
    for target_rank in range(21, min(42, best_rank)):
        if time.time() > deadline:
            break
        
        for attempt in range(200):  # up to 200 attempts per rank
            if time.time() > deadline:
                break
            
            # Generate random decomposition with the given rank
            U = [[rand_binary() for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rand_binary() for _ in range(target_rank)] for _ in range(m*p)]
            W = [[rand_binary() for _ in range(target_rank)] for _ in range(p*n)]
            
            # Check if this decomposes correctly
            if _is_valid(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        else:
            continue
        
        # If found at this rank, we could stop (we want minimal), but let's verify
        if best_rank <= target_rank:
            break
    
    # If no lower rank was found, return the schoolbook
    if best_rank == n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
