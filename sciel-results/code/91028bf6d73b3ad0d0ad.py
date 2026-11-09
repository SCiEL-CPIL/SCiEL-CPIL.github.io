"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _round_to_half(vals):
    """Round a list of floats to the nearest half-integer."""
    return [round(v * 2) / 2.0 for v in vals]

def _construct_rank18():
    """
    Construct a rank-18 decomposition for 2x3x5 matrix multiplication.
    
    Strategy:
    1. For j=0, we need to compute A[i,0]*B[0,k] for i in {0,1}, k in {0..4}.
       This is a 2x1 times 1x5 product. We can use 5 terms if we fix i, but we need both i.
       However, we can share the B[0,k] part? No, U depends on i.
       Let's use 5 terms where U[i,0,c] = 1, V[0,k,c] = 1, W[k,i,c] = 1? No, W index is (k,i).
       If we use 5 terms for j=0:
         c=0..4: U[i,0,c] = 1 for all i? No, U is (i,j) -> i*m+j.
         If U[i,0,c] = 1, V[0,k,c] = 1, W[k,i,c] = 1, then for fixed (i,k),
         sum_c U[i,0,c]V[0,k,c]W[k,i,c] = sum_{c: k_c=k} 1*1*1 = 1 if we have one term per k.
         But W[k,i,c] must be 1 for the specific i. If we have one term per k, we can only have W[k,i,c]=1 for one i.
         So we need 2 terms per k for j=0? That's 10 terms.
         
       Let's look at the structure again.
       T[i,j,k] = 1.
       
       Alternative:
       For j=0, use 5 terms.
       c=0..4.
       U[i,0,c] = 1.
       V[0,k,c] = 1 if k==c else 0.
       W[k,i,c] = 1.
       Then for (i,0,k): sum_c U[i,0,c]V[0,k,c]W[k,i,c] = U[i,0,k]V[0,k,k]W[k,i,k] = 1*1*1 = 1.
       This works for ALL i simultaneously because U and W are constant 1 for the relevant indices.
       Wait, U[i,0,c] is the element at row i*m+0, col c.
       If we set U[i,0,c] = 1 for all i and c in 0..4, then:
       Term c contributes U[i,0,c]*V[0,k,c]*W[k,i,c].
       If V[0,k,c] is 1 only when k=c, then for a fixed k, only term c=k contributes.
       The contribution is U[i,0,k]*V[0,k,k]*W[k,i,k] = 1*1*1 = 1.
       So 5 terms handle j=0 for all i and k.
       
       Similarly for j=1, we can use 5 terms?
       c=5..9.
       U[i,1,c] = 1.
       V[1,k,c] = 1 if k == c-5 else 0.
       W[k,i,c] = 1.
       Contribution for (i,1,k): 1.
       
       Similarly for j=2, we can use 5 terms?
       c=10..14.
       U[i,2,c] = 1.
       V[2,k,c] = 1 if k == c-10 else 0.
       W[k,i,c] = 1.
       Contribution for (i,2,k): 1.
       
       Total rank = 15.
       Let's verify.
       For any (i,j,k):
       Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
       
       If c is in [0,4]:
         U[i,0,c] is 1 only if j=0? No, U[i*m+j, c] is 0 if j != 0 because we only set U[i,0,c]=1.
         Wait, U is a matrix of size (n*m, rank).
         U[i*m+0, c] = 1 for c in 0..4.
         U[i*m+1, c] = 0 for c in 0..4.
         U[i*m+2, c] = 0 for c in 0..4.
         
         V[0*p+k, c] = 1 if k==c else 0.
         V[1*p+k, c] = 0.
         V[2*p+k, c] = 0.
         
         W[k*n+i, c] = 1.
         
         So for j=0:
         Term c=k contributes U[i,0,k]*V[0,k,k]*W[k,i,k] = 1*1*1 = 1.
         Other terms c!=k contribute 0 because V[0,k,c]=0.
         So sum = 1.
         
         For j=1:
         U[i,1,c] = 0 for c in 0..4.
         So sum = 0. Correct, because T[i,1,k] for j=1 should not be handled by j=0 terms.
         
       So this construction gives rank 15!
       
       Let's double check the indices.
       n=2, m=3, p=5.
       U shape: (6, 15).
       V shape: (15, 15).
       W shape: (10, 15).
       
       For c=0..4:
         U[i*3+0, c] = 1. (i=0,1)
         V[0*5+k, c] = 1 if k==c else 0.
         W[k*2+i, c] = 1. (k=0..4, i=0,1)
         
       For c=5..9:
         U[i*3+1, c] = 1.
         V[1*5+k, c] = 1 if k==(c-5) else 0.
         W[k*2+i, c] = 1.
         
       For c=10..14:
         U[i*3+2, c] = 1.
         V[2*5+k, c] = 1 if k==(c-10) else 0.
         W[k*2+i, c] = 1.
         
       Check T[0,0,0]:
       c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1.
       c=1..4: V[0,0,c]=0 -> 0.
       c=5..14: U[0,0,c]=0 -> 0.
       Sum = 1.
       
       Check T[0,1,2]:
       c=0..4: U[0,1,c]=0 -> 0.
       c=5: U[0,1,5]=1, V[1*5+2, 5] = V[7,5]. V[1,2,5] is 1 if 2==5-5=0? No. V[7,5] corresponds to j=1, k=2.
       V[j*p+k, c] = 1 if k == c-5.
       Here j=1, k=2, c=5. 2 == 5-5=0? False. So V[7,5]=0.
       Wait, c=7 corresponds to k=2 (since c-5=2).
       So for T[0,1,2], we need c=7.
       c=7: U[0,1,7]=1. V[1*5+2, 7] = V[7,7]. V[1,2,7] is 1 if 2==7-5=2. Yes. So V[7,7]=1.
       W[2*2+0, 7] = W[4,7] = 1.
       Product = 1.
       Other c in 5..9: V[7,c]=0.
       c in 0..4, 10..14: U[0,1,c]=0.
       Sum = 1.
       
       This looks correct. Rank 15 is better than 20.
    """
    n, m, p = 2, 3, 5
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U[i, j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j, k, c] = 1
            V[j * p + k][c] = 1
            # W[k, i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}

def _try_rank_r(r, deadline, seed=42):
    """
    Try to find a rank-r decomposition using random initialization and ALS,
    then snap to half-integers.
    """
    if r >= 15:
        return None # We already have 15
    
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(200):
        if time.time() > deadline - 5:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((r, p))
                for k in range(p):
                    for c in range(r):
                        A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((r, n))
                for i in range(n):
                    for c in range(r):
                        B[c, i] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((r, m))
                for j in range(m):
                    for c in range(r):
                        C[c, j] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    err += (actual - 1.0) ** 2
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-9:
        U_list = _round_to_half([x for row in best_U for x in row])
        V_list = _round_to_half([x for row in best_V for x in row])
        W_list = _round_to_half([x for row in best_W for x in row])
        
        U_arr = np.array(U_list).reshape(n*m, r)
        V_arr = np.array(V_list).reshape(m*p, r)
        W_arr = np.array(W_list).reshape(p*n, r)
        
        U_out = U_arr.tolist()
        V_out = V_arr.tolist()
        W_out = W_arr.tolist()
        
        if _verify(U_out, V_out, W_out, tol=1e-6):
            return {"U": U_out, "V": V_out, "W": W_out}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # First, check the rank-15 construction
    res15 = _construct_rank18() # Name is misleading, it's rank 15
    if _verify(res15['U'], res15['V'], res15['W']):
        best_result = res15
        best_rank = 15
    else:
        best_result = _schoolbook()
        best_rank = 30
        
    # Try to find lower rank decompositions (14, 13, ...)
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        result = _try_rank_r(target_rank, deadline, seed=target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
            
    return best_result
# EVOLVE-BLOCK-END
