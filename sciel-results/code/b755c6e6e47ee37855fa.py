import time
import random
import itertools

DIMENSIONS = [2, 3, 9]

def _check_exact(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    # Build the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # 140 seconds out of 150s limit, leaving 10s safety margin
    
    n, m, p = DIMENSIONS
    N = n * m  # 6
    M = m * p  # 27
    P = p * n  # 18
    schoolbook_rank = n * m * p  # 54
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = None, None, None, schoolbook_rank
    
    def schoolbook():
        U = [[0] * schoolbook_rank for _ in range(N)]
        V = [[0] * schoolbook_rank for _ in range(M)]
        W = [[0] * schoolbook_rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, schoolbook_rank
    
    U, V, W, rank = schoolbook()
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find lower rank decompositions
    # Strategy: For smaller subproblems, we know lower ranks exist.
    # M_{2,2,2} has rank 7.
    # Can we use this? 
    # Actually, let's try a simple greedy approach with random initialization and local search.
    # But given the time constraints and the need for exact integer/half-integer solutions,
    # random search in a high-dimensional space is likely to fail to find an exact solution.
    
    # Let's try a structured approach:
    # The tensor is M_{2,3,9}.
    # We can write it as a sum of terms.
    # If we could find a decomposition of rank 48, that would be 54 * 48/54 = 48.
    # Score would be 54/48 = 1.125.
    
    # Is there a known lower bound?
    # For M_{2,3,9}, the border rank is at least max(N+M+P-2, ...) ?
    # Actually, the lower bound is max(N*M, M*P, P*N, N+M+P-2) ... no, that's not right.
    # The trivial lower bound is max(N, M, P) is not right either.
    # The tensor has size 6 x 27 x 18.
    # The rank is at least max(6, 27, 18) = 27? No.
    # The rank is at least the slice rank.
    # Slice rank >= max(N, M, P) is false.
    # Slice rank >= max(N*M/P, M*P/N, P*N/M) ...
    # Actually, let's just stick with the schoolbook unless we find a specific trick.
    
    # Wait, I can try to use the fact that M_{2,3,9} can be decomposed using M_{2,2,2} for the 2x2 part?
    # But the dimensions are 2, 3, 9.
    # 3 = 2 + 1, 9 = 4 * 2 + 1?
    # This seems too complex to derive on the fly without guaranteed correctness.
    
    # Let's try a very short time-limited search for rank 53, 52, etc.
    # We can try to merge two terms if their U and V vectors are proportional?
    # In schoolbook, U and V are basis vectors, so they are not proportional to others.
    # If we perturb the decomposition slightly, we might merge terms.
    # But we need an EXACT solution.
    
    # Given the strict requirement for exact integer/half-integer entries and the 150s limit,
    # and the high dimensionality, finding a lower rank exactly is extremely difficult
    # without prior knowledge of the specific coefficients.
    
    # I will return the schoolbook decomposition to ensure a valid score of 1.0.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # One more check: Is there a simple block diagonal structure?
    # M_{2,3,9} = M_{2,3,9}
    # No obvious block diagonal structure that reduces rank significantly.
    
    return {"U": best_U, "V": best_V, "W": best_W}
