"""
Matrix multiplication tensor decomposition for 2x3 by 3x5 matrices.
Attempts to find a rank-minimal exact tensor decomposition.
Falls back to schoolbook (rank 30) if no lower rank is found.
"""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact using random integer inputs."""
    # Use a few random tests with small integers
    for _ in range(3):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B directly
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C[i][k] = s
        
        # Compute C using decomposition
        C_decomp = [[0] * p for _ in range(n)]
        for r in range(rank):
            # For each r, compute the contribution
            # a_part[i] = sum_j A[i,j] * U[i*m+j, r]
            # b_part[k] = sum_j B[j,k] * V[j*p+k, r]
            # C[i,k] += W[k*n+i, r] * a_part[i] * b_part[k]
            a_part = [0] * n
            b_part = [0] * p
            for i in range(n):
                for j in range(m):
                    a_part[i] += A[i][j] * U[i*m+j][r]
            for k in range(p):
                for j in range(m):
                    b_part[k] += B[j][k] * V[j*p+k][r]
            for i in range(n):
                for k in range(p):
                    C_decomp[i][k] += W[k*n+i][r] * a_part[i] * b_part[k]
        
        # Check exactness
        for i in range(n):
            for k in range(p):
                if C[i][k] != C_decomp[i][k]:
                    return False
    return True

def _try_rank_structured(target_rank, n, m, p, deadline, max_iters=200):
    """
    Try to find a decomposition of the given rank using structured approach.
    We try to use the structure of matrix multiplication.
    """
    start = time.time()
    
    # Initialize with random rational values
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random initialization with small rationals
        U = [[Fraction(random.randint(-3, 3), 1) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[Fraction(random.randint(-3, 3), 1) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[Fraction(random.randint(-3, 3), 1) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Quick check: compute tensor and see if it matches
        # The tensor T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For matrix multiplication, T[i,j,k] = 1 if this is a "term" A[i,j]*B[j,k]
        # Actually, the tensor for matrix multiplication is:
        # T[i, j, k] = 1 for all i,j,k (it's the all-ones tensor for the multiplication)
        # Wait, no. The tensor is such that:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) (sum_j B[j,k] V[j*p+k,r]) W[k*n+i,r]
        # So the tensor is: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta_{j, j} ... 
        # Actually, for the standard matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k.
        # Because C[i,k] = sum_j A[i,j] B[j,k], and the decomposition gives:
        # C[i,k] = sum_r W[k*n+i,r] * (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r])
        # = sum_j sum_r A[i,j] B[j,k] U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # So we need: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
        
        # Check a few entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(target_rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to integers if possible
            U_int = [[int(U[i][r]) if U[i][r].denominator == 1 else float(U[i][r]) for r in range(target_rank)] for i in range(n*m)]
            V_int = [[int(V[i][r]) if V[i][r].denominator == 1 else float(V[i][r]) for r in range(target_rank)] for i in range(m*p)]
            W_int = [[int(W[i][r]) if W[i][r].denominator == 1 else float(W[i][r]) for r in range(target_rank)] for i in range(p*n)]
            return (U_int, V_int, W_int)
    
    return None

def _try_rank_random(target_rank, n, m, p, deadline, max_iters=500):
    """
    Try random search for a decomposition of the given rank.
    """
    start = time.time()
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Check if valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return (U, V, W)
    
    return None

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    min_possible = max(n * m, m * p, p * n)
    
    for target_rank in range(min_possible, schoolbook_rank):
        if time.time() > deadline - 5:  # Stop searching if close to deadline
            break
        
        # Try structured search first
        result = _try_rank_structured(target_rank, n, m, p, deadline - 5, max_iters=100)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid decomposition, stop
        
        # Try random search
        result = _try_rank_random(target_rank, n, m, p, deadline - 5, max_iters=200)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid decomposition, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    # Verification logic
    result = construct()
    U = result['U']
    V = result['V']
    W = result['W']
    
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Verify with random matrices
    random.seed(42)
    A = [[random.randint(1, 10) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(1, 10) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C[i][k] = s
    
    # Compute C using decomposition
    C_decomp = [[0] * p for _ in range(n)]
    for r in range(rank):
        a_part = [0] * n
        b_part = [0] * p
        for i in range(n):
            for j in range(m):
                a_part[i] += A[i][j] * U[i*m+j][r]
        for k in range(p):
            for j in range(m):
                b_part[k] += B[j][k] * V[j*p+k][r]
        for i in range(n):
            for k in range(p):
                C_decomp[i][k] += W[k*n+i][r] * a_part[i] * b_part[k]
    
    # Check if C == C_decomp
    is_valid = True
    for i in range(n):
        for k in range(p):
            if C[i][k] != C_decomp[i][k]:
                is_valid = False
                print(f"Mismatch at {i},{k}: {C[i][k]} != {C_decomp[i][k]}")
    
    if is_valid:
        print(f"Valid decomposition with rank {rank}")
    else:
        print("Invalid decomposition")
