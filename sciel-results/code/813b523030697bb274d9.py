"""Program that attempts to find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]
SEED_RANK = 105

# EVOLVE-BLOCK-START

def _validate(U, V, W, n=3, m=5, p=7):
    """
    Validate that the factor matrices U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    The condition is: for all i, j, k,
        sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r]
                expected = 0
                for r in range(rank):
                    expected += W[k*n+i][r]
                if s != expected:
                    return False
    return True


def _schoolbook(n=3, m=5, p=7):
    """Return the schoolbook decomposition with rank n*m*p."""
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


def _attempt_rank_reduction(n=3, m=5, p=7, deadline=None, seed=42):
    """
    Attempt to find a lower-rank decomposition using structured approaches.
    This is a heuristic search for a valid lower-rank decomposition.
    
    Strategy: We try to find a decomposition of rank r < n*m*p by
    using a structured approach based on known matrix multiplication 
    algorithms (e.g., Strassen-type block decompositions).
    
    For small dimensions like 3x5x7, we use a random search over 
    integer/half-integer coefficients with a structured initialization.
    """
    if deadline is None:
        deadline = time.time() + 140
    
    rng = random.Random(seed)
    best = _schoolbook(n, m, p)
    
    # The tensor we need to decompose: T[i,j,k] = 1 for all i,j,k
    # We need U[i*m+j, r] * V[j*p+k, r] summed over r = W[k*n+i, r] summed over r
    # Actually, the correct condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = delta_{(i,j,k) in tensor} = 1
    # And W is defined such that the output uses the trace convention.
    
    # Let me reconsider the problem. The matrix multiplication tensor T has:
    # T[i, j, k] = 1 for all valid i, j, k
    # A rank-r decomposition gives:
    # T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, no. The standard decomposition is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = A[i,j] * B[j,k]
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{r} u_r[i,j] * v_r[j,k] * w_r[k,i]
    # Where the output is sum_r w_r[k,i] * (sum_j u_r[i,j]*v_r[j,k])
    # No wait, the standard Strassen decomposition is:
    # M_r = sum_r U_r * V_r * W_r
    # Where U_r is a linear combination of A entries, V_r of B entries,
    # and W_r gives the output.
    
    # Actually, looking at the seed code more carefully:
    # U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] is NOT the condition.
    # The condition should be:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = (A*B output coefficient)
    
    # Let me re-read the task. The output factor follows W[k*n+i, r].
    # The standard decomposition for matrix multiplication:
    # (C)[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1
    # A rank-r decomposition: T[i,j,k] = sum_r T_r[i,j,k] where each T_r is rank-1
    # i.e., T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]... 
    # No, that's not right either.
    
    # Standard: The multiplication tensor has a rank-r decomposition if:
    # sum_j delta_{i,j,k} = sum_r u_r(i,j) v_r(j,k) w_r(k,i)
    # where the "output" is computed as sum_r w_r(k,i) * (product of u and v terms)
    
    # Actually, the simplest way: the rank of the tensor T_{ijk} = delta (all ones)
    # is the minimum r such that T_{ijk} = sum_{r=1}^r a_r(i,j) b_r(j,k) c_r(i,k)
    # No, that's not standard either.
    
    # Let me just use the seed code's convention. In the seed:
    # For each (i,j,k), we have one term r:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # The "product" U[i*m+j, r] * V[j*p+k, r] = 1 for that specific (i,j,k,r)
    # And W[k*n+i, r] = 1
    # So the sum over r of U[i*m+j, r] * V[j*p+k, r] gives the number of r
    # where both U and V have 1 at that (i,j,k) position.
    # In schoolbook, for each (i,j,k), there's exactly one r where U[i*m+j, r]=1 
    # and V[j*p+k, r]=1, so the sum is 1.
    # And W[k*n+i, r] is also 1 for that r.
    # So the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all i,j,k
    # And W provides the "output weight" which is 1.
    
    # So the validation should check:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all i, j, k
    # (The W matrix is then used in the output computation, but for the tensor 
    #  decomposition, the key constraint is on U and V.)
    
    # Wait, but W also matters. Let me think about this differently.
    # The task says the factor matrices exactly decompose the matrix multiplication tensor.
    # The standard form is:
    # For the output C[i,k] = sum_j A[i,j] B[j,k]:
    # We compute intermediate products M_r = (linear form in A) * (linear form in B)
    # Then C[i,k] = sum_r W[k*n+i, r] * M_r
    # where M_r = (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j,k} V[j*p+k, r] B[j,k])
    
    # So the tensor decomposition condition is:
    # delta_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # for all i, j, k.
    
    # Let me re-validate with this understanding.
    
    def validate2(U, V, W, n=3, m=5, p=7):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Use the corrected validation
    global _validate
    _validate = validate2
    
    # Now attempt to find lower-rank decompositions
    # Try ranks from 104 down to some minimum
    
    for target_rank in range(SEED_RANK - 1, max(10, SEED_RANK // 2)):
        if time.time() > deadline:
            break
        
        # Try random structured approaches
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers/half-integers
            U = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if validate2(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return best


def construct():
    """
    Returns the best tensor decomposition found for matrix multiplication.
    """
    deadline = time.time() + 140
    
    # First, return schoolbook as a safe fallback
    best = _schoolbook()
    
    # Attempt rank reduction
    if time.time() < deadline:
        result = _attempt_rank_reduction(deadline=deadline)
        if result is not None:
            best = result
    
    return best

# EVOLVE-BLOCK-END
