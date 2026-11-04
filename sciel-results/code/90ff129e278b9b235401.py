import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Construct rank-49 decomposition for 2x5x5 matrix multiplication.
    
    Known result: For 2xn x n, rank 2n^2 - 1 is achievable.
    For n=5: rank = 2*25 - 1 = 49.
    
    Construction: Use 48 schoolbook terms for all (i,j,k) except (i=0,j=4,k=4),
    plus 1 additional term that compensates. The key identity:
    A[0,4]*B[4,4] can be expressed using the other products in the last row/col.
    
    More precisely, we use the fact that the tensor has a linear relation:
    For the output C[0,4], we need sum_j A[0,j]*B[j,4].
    We keep schoolbook terms for j=0,1,2,3 and modify the j=4 term.
    Similarly for C[1,4].
    The saving comes from combining the two j=4,k=4 terms into one.
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    n, m, p = 2, 5, 5
    target_rank = 49
    
    try:
        # Use a structured construction.
        # We'll have 49 terms indexed c=0..48.
        # Terms c=0..47: schoolbook for 48 of the 50 (i,j,k) triples.
        # Term c=48: a correction term.
        
        # Which (i,j,k) to omit? Let's omit (i=0, j=4, k=4).
        # We need to add contributions to all outputs that depended on this term.
        # The term (i=0,j=4,k=4) contributes to C[0,4] via A[0,4]*B[4,4].
        
        # But a single rank-1 term can only affect outputs where all three
        # factors are nonzero. So we need a more careful construction.
        
        # Alternative approach: Use the known explicit construction for 2xn x n.
        # The construction from Bini, De Falco, Roatta (1981) or later work.
        
        # For 2x2x2, rank 7:
        # The 7 terms are:
        # c=0: U[0,0]=1,V[0,0]=1,W[0,0]=1  -> A00*B00 -> C00
        # c=1: U[0,1]=1,V[1,0]=1,W[0,1]=1  -> A01*B10 -> C01  
        # c=2: U[1,0]=1,V[0,1]=1,W[1,0]=1  -> A10*B01 -> C10
        # c=3: U[1,1]=1,V[1,1]=1,W[1,1]=1  -> A11*B11 -> C11
        # c=4: U[0,0]=1,V[0,1]=1,W[1,0]=1  -> A00*B01 -> C10 (extra)
        # c=5: U[1,0]=1,V[0,0]=1,W[0,1]=1  -> A10*B00 -> C01 (extra)
        # c=6: U[0,1]=1,V[1,0]=1,W[1,1]=1  -> A01*B10 -> C11 (extra)
        # ... this is getting complicated.
        
        # Let me use a cleaner known construction.
        # For 2xn x n, the rank 2n^2-1 decomposition can be built as follows:
        # 
        # For each (j,k) in {0..n-1}^2:
        #   If (j,k) != (n-1, n-1):
        #     Term for i=0: U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
        #     Term for i=1: U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
        #   If (j,k) == (n-1, n-1):
        #     One combined term using auxiliary structure
        #
        # But this gives 2*(n^2-1) + 1 = 2n^2-1 terms only if the combined
        # term correctly produces both C[0,n-1] and C[1,n-1] contributions.
        # A single rank-1 term U[i,j]*V[j,k]*W[k,i] produces exactly one
        # output element. So we can't combine two different outputs.
        
        # The correct construction must be different. Let me look at this
        # from the tensor rank perspective.
        
        # The matrix multiplication tensor for 2xn x n has a specific structure.
        # The rank 2n^2-1 result comes from the fact that the tensor lies in
        # a subspace of dimension 2n^2-1 in some appropriate flattening.
        
        # Let me try a direct numerical search with rounding.
        np.random.seed(12345)
        
        # Start from schoolbook and perturb
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Fill first 49 columns with schoolbook-like structure
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if c >= target_rank:
                        break
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        
        # Now we have 49 schoolbook terms (missing the last one: i=1,j=4,k=4)
        # The missing term was: U[1*5+4]=U[9], V[4*5+4]=V[24], W[4*2+1]=W[9]
        # We need to modify the existing terms to compensate.
        
        # This approach is too simplistic. Let me use gradient descent on
        # the full 49-term decomposition.
        
        # Initialize: 48 schoolbook terms + 1 random term
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) == (1, 4, 4):  # skip this one
                        continue
                    if c >= target_rank - 1:
                        break
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        
        # Use a structured construction based on the known rank 2n^2-1 result.
        # The construction: for each (j,k) pair except (n-1,n-1), use 2 schoolbook terms.
        # For (j,k)=(n-1,n-1), use 1 term with carefully chosen coefficients.
        #
        # The single term for (j=n-1,k=n-1) must produce:
        #   contribution to C[0,n-1]: A[0,n-1]*B[n-1,n-1]
        #   contribution to C[1,n-1]: A[1,n-1]*B[n-1,n-1]
        #
        # But a rank-1 term U[i,j]*V[j,k]*W[k,i] only produces ONE output.
        # So this simple approach doesn't work directly.
        #
        # The correct construction modifies multiple terms. Let me use a
        # different strategy: perturb the schoolbook decomposition.
        
        # Start with full schoolbook (50 terms) and try to merge two terms.
        # We want to find U,V,W with 49 columns such that the tensor equation holds.
        
        # Use the fact that we can write the missing product as a linear
        # combination of other products plus a new term.
        
        # Let's try: express A[0,4]*B[4,4] using other available products.
        # Available: A[0,j]*B[j,4] for j=0,1,2,3 and A[1,j]*B[j,4] for j=0,1,2,3,4
        #            A[0,4]*B[j,4] for j=0,1,2,3 (already counted)
        #            A[1,4]*B[4,4]
        #
        # This doesn't give a linear relation in general.
        
        # Let me try a completely different approach: use the explicit
        # construction from the literature for 2xn x n rank 2n^2-1.
        
        # The construction uses n^2 - 1 "standard" terms for each row,
        # plus 1 shared term, giving 2(n^2-1) + 1 = 2n^2-1 total.
        
        # For row i, column k: C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # For (j,k) != (n-1,n-1):
        #   Term_{i,j,k}: U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # For (j,k) = (n-1,n-1):
        #   We need to produce A[i,n-1]*B[n-1,n-1] for i=0,1
        #   Use a shared term with auxiliary structure
        
        # Actually, the standard construction for 2x2x2 rank 7 is:
        # c=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  (A00*B00->C00)
        # c=1: U=[1,0,0,0], V=[0,1,0,0], W=[0,1,0,0]  (A00*B01->C01)  
        # c=2: U=[0,1,0,0], V=[1,0,0,0], W=[0,0,1,0]  (A10*B00->C10)
        # c=3: U=[0,1,0,0], V=[0,1,0,0], W=[0,0,0,1]  (A11*B11->C11)
        # c=4: U=[1,0,0,0], V=[0,0,1,0], W=[0,0,1,0]  (A00*B10->C10 extra)
        # c=5: U=[0,0,1,0], V=[1,0,0,0], W=[0,1,0,0]  (A01*B00->C01 extra)
        # c=6: U=[0,0,1,0], V=[0,0,1,0], W=[0,0,0,1]  (A01*B10->C11 extra)
        
        # Let me verify c=4: contributes to C[1,0] via U[1,0]=1? No, U[1,0] is for i=1,j=0.
        # U[1,0]=1 means this term uses A[1,0]. But we want A[0,0]*B[1,0].
        # So U should have U[0,0]=1 for A[0,0]. Let me recheck.
        
        # I'm getting confused with indices. Let me just try multiple random
        # initializations with gradient descent and see if any converge.
        
        for attempt in range(5):
            if time.time() > _DEADLINE - 15:
                break
            np.random.seed(1000 + attempt)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with schoolbook for first 49 terms
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if c >= target_rank:
                            break
                        U[i*m+j, c] = 1
                        V[j*p+k, c] = 1
                        W[k*n+i, c] = 1
                        c += 1
            
            # Add small random perturbation to all terms
            U += np.random.randn(n*m, target_rank) * 0.01
            V += np.random.randn(m*p, target_rank) * 0.01
            W += np.random.randn(p*n, target_rank) * 0.01
            
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for cc in range(target_rank):
                                s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                            err += (s - 1.0)**2
                return err
            
            def gradients(U, V, W):
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for cc in range(target_rank):
                                s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                            ds = 2.0 * (s - 1.0)
                            for cc in range(target_rank):
                                gU[i*m+j, cc] += ds * V[j*p+k, cc] * W[k*n+i, cc]
                                gV[j*p+k, cc] += ds * U[i*m+j, cc] * W[k*n+i, cc]
                                gW[k*n+i, cc] += ds * U[i*m+j, cc] * V[j*p+k, cc]
                return gU, gV, gW
            
            lr = 0.005
            for step in range(3000):
                if time.time() > _DEADLINE - 15:
                    break
                gU, gV, gW = gradients(U, V, W)
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                if step % 500 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-12:
                        break
            
            err = compute_error(U, V, W)
            if err > 1e-6:
                continue
            
            # Try rounding
            for scale in [1, 2]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                U_list = [[float(U_r[i, j]) for j in range(target_rank)] for i in range(n*m)]
                V_list = [[float(V_r[i, j]) for j in range(target_rank)] for i in range(m*p)]
                W_list = [[float(W_r[i, j]) for j in range(target_rank)] for i in range(p*n)]
                if _verify(n, m, p, U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
        
    except Exception:
        return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
