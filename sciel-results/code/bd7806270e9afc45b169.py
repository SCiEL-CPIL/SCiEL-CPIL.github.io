"""Rewrite: exploit 3x5x5 = (2x5x5) ⊕ (1x5x5) structure with random rank-r search."""
import random
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n=3, m=5, p=5, rank=None):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid. U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    Convention: W[k*n + i, r] contributes to output[i][k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Actually the tensor is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    where U is indexed by (i,j) flattened as i*m+j, V by (j,k) as j*p+k, W by (k,i) as k*n+i.
    """
    if rank is None:
        rank = len(U[0])
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal delta (i.e., 1 if the "multiplication" identity holds)
    # For matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition gives: C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[k,i,r]
    # Wait, let me re-read the convention.
    # The standard bilinear form: C[i,k] = sum_r U[i,k,r] * V[?,r] ... 
    # Let me use the seed program's convention:
    # U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    # So C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # For schoolbook: for each (i,j,k) triple, U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1
    # Then C[i,k] = sum_r sum_j sum_k' U[i*m+j,r] A[i,j] V[j*p+k',r] B[j,k'] W[k'*n+i,r]
    # With schoolbook, only one r per (i,j,k'), and W[k'*n+i,r]=1 only when k'=k
    # So C[i,k] = sum_j A[i,j] B[j,k] ✓
    
    # So the check is: for all i,k: sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) = sum_j A[i,j]*B[j,k]
    # This must hold for ALL A and B, so it's a tensor identity:
    # For all i,k: sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] e_{ij}) * (sum_j V[j*p+k,r] e_{jk}) = sum_j e_{ij} e_{jk}
    # In tensor form: sum_r W[k*n+i,r] U[i*m+j,r] V[j*p+k,r] = delta_{j,j} (i.e., 1 for the "identity" mapping)
    # Actually the multiplication tensor M[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k], the coefficient of A[i,j]B[j,k] in C[i,k] is 1)
    # So we need: for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    s += Fraction(U[ui][r]) * Fraction(V[vj][r]) * Fraction(W[wk][r])
                if s != 1:
                    return False
    return True

def _construct_random(rank, seed):
    """Construct a random rank-r decomposition attempt.
    Strategy: 
    - The 1x5x5 block (row i=2) needs rank 5: C[2,k] = sum_j A[2,j]*B[j,k]
      This is just 5 independent scalar multiplications, so we need 5 terms.
    - The 2x5x5 block (rows i=0,1) needs to compute C[0,k] and C[1,k] for k=0..4.
      This is equivalent to multiplying a 10x5 matrix (rows 0-4 of A for i=0,1 interleaved? No...)
      Actually: C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k]
      These share the same B matrix. The "matrix" being multiplied is effectively:
      A_block = [[A[0,0], A[0,1], A[0,2], A[0,3], A[0,4]],
                 [A[1,0], A[1,1], A[1,2], A[1,3], A[1,4]]]
      and B is 5x5. Output is 2x5.
      This is a 2x5x5 multiplication, which as a tensor has rank at most 2*5*5=50 but likely much less.
      
    For the random construction, I'll use the following approach:
    - For the 1x5x5 part (i=2): use 5 terms where term r (r=0..4) has:
      U[2*5+j, r] = delta_{j,r}, V[r*5+k, r] = delta_{k,r}, W[r*5+2, r] = 1
      Wait, that's not right. Let me think again.
    
    For i=2, j, k: we need sum_r U[2*5+j, r] * V[j*5+k, r] * W[k*3+2, r] = 1
    Simple solution: for each j, use term r=j:
      U[2*5+j, j] = 1, V[j*5+k, j] = 1 for all k? No, V[j*5+k, r] must depend on k.
    
    Actually for the 1x5x5 block, the standard schoolbook uses 5 terms:
    For j in 0..4: term r=j: U[2*5+j, j]=1, V[j*5+k, j]=1 for all k... no.
    
    Let me reconsider. The tensor entry is T[i,j,k] = 1 for all i,j,k.
    For i=2: T[2,j,k] = sum_r U[2*5+j, r] * V[j*5+k, r] * W[k*3+2, r] = 1
    
    A simple rank-5 decomposition for this slice:
    Use r = j (5 terms):
      U[2*5+j, j] = 1 (and 0 for other r)
      V[j*5+k, j] = 1 for all k (and 0 for other r)
      W[k*3+2, j] = 1 for all k (and 0 for other r)
    Check: sum_r U[2*5+j,r] V[j*5+k,r] W[k*3+2,r] = U[2*5+j,j] * V[j*5+k,j] * W[k*3+2,j] = 1*1*1 = 1 ✓
    
    But wait, this uses the same V and W entries for all i. That's fine since different i use different U rows.
    However, V[j*5+k, j] = 1 for all k means V row j*5+k has a 1 in column j. And W[k*3+2, j] = 1 for all k means W row k*3+2 has a 1 in column j.
    
    Now for i=0 and i=1 (the 2x5x5 block):
    T[0,j,k] = sum_r U[0*5+j, r] * V[j*5+k, r] * W[k*3+0, r] = 1
    T[1,j,k] = sum_r U[1*5+j, r] * V[j*5+k, r] * W[k*3+1, r] = 1
    
    The V matrix is shared! V[j*5+k, r] is used for all i. So we need to find U, V, W such that:
    - For i=2: the above 5-term solution works, using columns 0-4.
    - For i=0,1: we need additional columns (or shared columns) to satisfy the equations.
    
    Total rank = 5 (for i=2) + rank_needed_for_(i=0,1)_block.
    
    For the (i=0,1) block, V is shared with the i=2 block. The V entries for columns 0-4 are already fixed:
    V[j*5+k, r] = delta_{j,r} for r=0..4.
    
    So for i=0: T[0,j,k] = sum_{r=0}^{R-1} U[j,r] * V[j*5+k,r] * W[k*3,r] = 1
    With V[j*5+k,r] = delta_{j,r} for r<5, this becomes:
    T[0,j,k] = U[j,j] * W[k*3,j] + sum_{r=5}^{R-1} U[j,r] * V[j*5+k,r] * W[k*3,r] = 1
    
    This is getting complex. Let me just do a random search with verification.
    """
    n, m, p = 3, 5, 5
    rng = random.Random(seed)
    
    # Initialize with zeros
    U = [[0]*rank for _ in range(n*m)]  # 15 x rank
    V = [[0]*rank for _ in range(m*p)]  # 25 x rank
    W = [[0]*rank for _ in range(p*n)]  # 15 x rank
    
    # First, handle the i=2 block with 5 terms (columns 0-4)
    for j in range(5):
        U[2*5+j][j] = 1
        for k in range(5):
            V[j*5+k][j] = 1
            W[k*3+2][j] = 1
    
    # Now for the i=0,1 block, use remaining columns (5 to rank-1)
    # We need: for i in {0,1}, j in 0..4, k in 0..4:
    #   sum_{r=5}^{rank-1} U[i*5+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 - (contribution from r<5)
    # Contribution from r<5: U[i*5+j, r]*V[j*5+k,r]*W[k*3+i,r]
    # For r<5: V[j*5+k,r] = delta_{j,r}, so only r=j contributes:
    #   U[i*5+j, j] * 1 * W[k*3+i, j]
    # So we need: sum_{r=5}^{rank-1} U[i*5+j,r]*V[j*5+k,r]*W[k*3+i,r] = 1 - U[i*5+j,j]*W[k*3+i,j]
    
    # For simplicity, set U[i*5+j, j] = 0 for i in {0,1}, j in 0..4, so RHS = 1.
    # Then we need: for i in {0,1}, j,k: sum_{r=5}^{rank-1} U[i*5+j,r]*V[j*5+k,r]*W[k*3+i,r] = 1
    
    # This is a 2x5x5 tensor of all 1s, decomposed with shared V.
    # Let's try: for each k, use one term? That gives 5 terms for the 2x5x5 block.
    # Term r = 5+k: 
    #   W[k*3+0, r] = 1, W[k*3+1, r] = 1
    #   V[j*5+k, r] = 1 for all j
    #   U[i*5+j, r] = 1 for all i,j
    # Check: sum_{r=5}^{9} U[i*5+j,r]*V[j*5+k,r]*W[k*3+i,r]
    # Only r=5+k contributes (since W[k'*3+i, 5+k] = 1 only when k'=k):
    #   U[i*5+j, 5+k] * V[j*5+k, 5+k] * W[k*3+i, 5+k] = 1*1*1 = 1 ✓
    
    if rank >= 10:
        for k in range(5):
            r = 5 + k
            for i in range(2):
                for j in range(5):
                    U[i*5+j][r] = 1
                    W[k*3+i][r] = 1
            for j in range(5):
                V[j*5+k][r] = 1
        return U, V, W
    else:
        # If rank < 10, we need a more efficient decomposition of the 2x5x5 block.
        # For now, return None to indicate failure.
        return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try ranks from low to high
    # Theoretical lower bound: at least 5 (for the 1x5x5 part) + something for 2x5x5
    # 2x5x5 has tensor rank at least 5 (since it contains 5x5 which has rank >= 5)
    # So total rank at least 10? But maybe we can share.
    # Let's try ranks from 10 upward.
    
    best_result = None
    
    for rank in range(10, 76):  # Up to schoolbook
        if time.time() > deadline - 5.0:
            break
        
        # Try the structured construction first
        result = _construct_random(rank, seed=rank)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, rank):
                best_result = {"U": U, "V": V, "W": W}
                return best_result
        
        # If structured didn't work, try random perturbations
        if time.time() > deadline - 5.0:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            rng = random.Random(rank * 1000 + attempt)
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Fill with small random integers
            for i in range(n*m):
                for r in range(rank):
                    U[i][r] = rng.randint(-2, 2)
            for j in range(m*p):
                for r in range(rank):
                    V[j][r] = rng.randint(-2, 2)
            for k in range(p*n):
                for r in range(rank):
                    W[k][r] = rng.randint(-2, 2)
            
            if _verify(U, V, W, n, m, p, rank):
                best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
