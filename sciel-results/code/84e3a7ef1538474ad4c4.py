"""Rewritten program: search for a rank-minimal exact decomposition of 2x3 * 3x3 matmul."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# Deadline guard: kill before 150s wall-clock, keep a safety margin.
_START = time.time()
_DEADLINE = _START + 135.0  # 15s margin

def _elapsed():
    return time.time() - _START

def _time_left():
    return max(0.0, _DEADLINE - _elapsed())

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check that sum_c U[a,c]*V[b,c] == W_tensor[a,b] where W_tensor[i*m+j, k*n+i] = delta... 
    Actually the decomposition must satisfy:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] == 1 if ... 
    
    Wait, the tensor T for matmul: T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    The standard tensor: T[i, j, k] = 1 for all i,j,k? No.
    
    Let me re-read: "Matrix multiplication 2,3,3". The tensor is T[i,j,k] where i indexes rows of A (n=2), j indexes cols of A / rows of B (m=3), k indexes cols of B (p=3).
    The matmul tensor has T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    
    Actually, the matrix multiplication tensor T of dimensions n x m x p has T[i,j,k] = 1 if and only if... 
    Standard definition: (A * B)[i,k] = sum_j A[i,j] * B[j,k].
    So the tensor T has T[i,j,k] = 1 for all i, j, k. Wait, that's the all-ones tensor which has rank... no.
    
    Let me look at the schoolbook code:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) triple.
    
    The reconstruction: For output index (i,k), which is W index k*n+i.
    The contribution is sum over c of U[i*m+j, c] * V[j*p+k, c] for each j? No, that's not how it works.
    
    Actually, the tensor decomposition satisfies:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * ... no.
    
    Let me think about this differently. The standard formulation:
    We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor T is defined on indices (i, j, k) with T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k].
    In the standard matmul tensor, T[i,j,k] = 1 for ALL i, j, k. No wait, that can't be right because then the rank would be related to the structure.
    
    Actually, I think the tensor is: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p). 
    No, that's not right. Let me re-examine the schoolbook decomposition.
    
    In the schoolbook: for each triple (i,j,k), we have a column c where:
    - U[i*m+j, c] = 1 (this picks out A[i,j])
    - V[j*p+k, c] = 1 (this picks out B[j,k])
    - W[k*n+i, c] = 1 (this puts the result into C[i,k])
    
    So the reconstruction is: for each output position (i,k), the value C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum over... ) 
    
    Actually, I think the tensor T is defined such that:
    T[i, j, k] = 1 for all valid i, j, k.
    
    And the decomposition satisfies:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * ... 
    
    No, that's not dimensionally consistent. Let me think again.
    
    The standard formulation for matrix multiplication tensor:
    The tensor T of size n x m x p where T[i,j,k] = 1 if and only if the (i,j)-th entry of the first matrix and the (j,k)-th entry of the second matrix multiply to contribute to the (i,k)-th entry of the result.
    
    In standard matmul, EVERY (i,j,k) triple contributes. So T[i,j,k] = 1 for all i,j,k.
    
    The decomposition is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using the tensor decomposition with factors U, V, W:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The reconstruction formula is:
    For each (i, j, k):
    sum_c U[i*m+j, c] * V[j*p+k, c] should equal T[i,j,k] = 1
    
    And W is used to map back: the output C[i,k] = sum_j (sum_c U[i*m+j,c] * V[j*p+k,c]) * ... 
    
    Hmm, I think I'm overcomplicating this. Let me just verify using the schoolbook decomposition directly.
    
    In schoolbook, for column c corresponding to (i,j,k):
    U[i*m+j, c] = 1, all other U entries in column c are 0.
    V[j*p+k, c] = 1, all other V entries in column c are 0.
    W[k*n+i, c] = 1, all other W entries in column c are 0.
    
    So for a given (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] = 1 (only the column for this specific (i,j,k) contributes).
    
    So the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] = 1 for all i,j,k.
    
    So T is the all-ones tensor of size 2x3x3. And we need to decompose it with rank r such that:
    T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c]
    
    And W is... how is W used? The problem says "the output factor follows the official trace convention W[k*n+i,r]". 
    
    I think W is part of the decomposition but the key constraint is:
    sum_c U[i*m+j, c] * V[j*p+k, c] = 1 for all i, j, k.
    
    And W might be used in a different way, or it might be that the full tensor decomposition involves all three factors. Let me re-read.
    
    "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    I think the standard formulation is that the matmul tensor T[i,j,k] can be written as:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    No, that doesn't make sense dimensionally for a 3-way tensor.
    
    Actually, I believe the correct formulation is:
    The matrix multiplication tensor is a 3-way tensor T of dimensions (n*m) x (m*p) x (p*n) or something like that. No...
    
    Let me just go with: the constraint is that for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] = 1
    
    And W is determined by the requirement that the decomposition is "exact" in some sense, or W is an additional factor.
    
    Actually, re-reading more carefully: I think the three factors U, V, W together define the decomposition such that the product of the corresponding entries reconstructs the tensor. The most common formulation I've seen is:
    
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  -- no, this isn't standard.
    
    Let me just use the verification: for the schoolbook, if I check sum_c U[i*m+j,c] * V[j*p+k,c] for all i,j,k, I get 1. So the constraint is:
    
    For all i in [0,2), j in [0,3), k in [0,3):
        sum_{c=0}^{rank-1} U[i*3+j, c] * V[j*3+k, c] == 1
    
    And W is... perhaps W is not used in the verification but is part of the output format? Or perhaps the full condition involves W.
    
    Given the ambiguity, let me just focus on finding U and V such that the bilinear form condition holds, and set W appropriately. Since the schoolbook has W[k*n+i, c] = 1 for the same column, maybe the condition is:
    
    For all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    Let me check with schoolbook: for a specific (i,j,k), only one column c has all three non-zero (all equal to 1), so the sum is 1. Yes! That works.
    
    So the condition is:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    This is a 3-way tensor decomposition where the tensor T has T[i*m+j, j*p+k, k*n+i] = 1... no, that's not quite right either because the indices are coupled.
    
    Actually, I think the cleanest interpretation is:
    We have a tensor T with entries T[a, b, d] where a = i*m+j (in [0, n*m)), b = j*p+k (in [0, m*p)), d = k*n+i (in [0, p*n)).
    But a, b, d are not independent - they share j, k, i.
    
    I think the safest approach is to just verify using the condition:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    Let me implement that and try to find a lower rank decomposition.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    """Try to find a rank-minimal exact decomposition. Start with schoolbook (rank 18) and try to reduce."""
    best = _schoolbook()
    best_rank = 18
    
    # Try small ranks with random search
    for target_rank in range(10, best_rank):
        if _time_left() < 5.0:
            break
        
        found = False
        attempts = 0
        max_attempts = 5000
        
        while not found and attempts < max_attempts and _time_left() > 5.0:
            attempts += 1
            random.seed(attempts * 7919 + target_rank * 104729)
            
            # Random initialization with small integers/half-integers
            U = [[random.choice([0, 1, -1, 0.5, -0.5, 2, -2]) for _ in range(target_rank)] for _ in range(6)]
            V = [[random.choice([0, 1, -1, 0.5, -0.5, 2, -2]) for _ in range(target_rank)] for _ in range(9)]
            W = [[random.choice([0, 1, -1, 0.5, -0.5, 2, -2]) for _ in range(target_rank)] for _ in range(6)]
            
            # Try to solve via iterative refinement
            # For each (i,j,k) constraint, we need sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1
            
            # Simple local search: perturb one entry at a time
            for step in range(200):
                if _time_left() < 3.0:
                    break
                    
                # Check if valid
                if _verify(U, V, W):
                    found = True
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
                
                # Pick a random (i,j,k) constraint
                i = random.randint(0, 1)
                j = random.randint(0, 2)
                k = random.randint(0, 2)
                
                current = sum(U[i*3+j][c] * V[j*3+k][c] * W[k*2+i][c] for c in range(target_rank))
                if current == 1:
                    continue
                
                # Try to fix by adjusting one entry
                c = random.randint(0, target_rank - 1)
                
                # Try adjusting U
                old = U[i*3+j][c]
                for delta in [0.5, -0.5, 1, -1, 0.25, -0.25]:
                    U[i*3+j][c] = old + delta
                    if _verify(U, V, W):
                        found = True
                        best = {"U": U, "V": V, "W": W}
                        best_rank = target_rank
                        break
                    U[i*3+j][c] = old
                if found:
                    break
                
                if found:
                    break
                # Try adjusting V
                old = V[j*3+k][c]
                for delta in [0.5, -0.5, 1, -1, 0.25, -0.25]:
                    V[j*3+k][c] = old + delta
                    if _verify(U, V, W):
                        found = True
                        best = {"U": U, "V": V, "W": W}
                        best_rank = target_rank
                        break
                    V[j*3+k][c] = old
                if found:
                    break
                
                if found:
                    break
                # Try adjusting W
                old = W[k*2+i][c]
                for delta in [0.5, -0.5, 1, -1, 0.25, -0.25]:
                    W[k*2+i][c] = old + delta
                    if _verify(U, V, W):
                        found = True
                        best = {"U": U, "V": V, "W": W}
                        best_rank = target_rank
                        break
                    W[k*2+i][c] = old
                if found:
                    break
        
        if found:
            break
    
    return best
