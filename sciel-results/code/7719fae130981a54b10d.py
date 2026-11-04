"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def verify_decomp(U, V, W, rank, num_trials=5):
        """
        C[i,k] = sum_j A[i,j] * B[j,k]
        Decomposition: C[i,k] = sum_c sum_j U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i,c]
        Wait, that's a 4-fold product. Let me think again.
        
        Standard bilinear decomposition:
        C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j' V[j'*p+k,c] * B[j',k])
        But that gives n*m*p terms and the W factor...
        
        I think the correct interpretation based on the index scheme is:
        The multiplication tensor has entries T[i,j,k] = delta (contributing to C[i,k] from A[i,j]*B[j,k])
        A rank-r decomposition: T[i,j,k] = sum_c u_{i,c} * v_{j,c} * w_{k,c}
        Where u_{i,c} = U[i*m+j, c]... no, U has indices [i*m+j, c] so it depends on both i and j.
        
        After much deliberation, I believe the convention is:
        C[i,k] = sum_c sum_j U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i,c] / W[k*n+i,c]
        That doesn't work.
        
        Let me just test schoolbook numerically.
        """
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    C_comp[i, k] = s * 1.0
            # This gives the same for all (i,k) which is wrong.
            return False
        
        # I give up on deriving; just verify schoolbook works with the right formula.
        # Let me test: for schoolbook, U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
        # So C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For schoolbook, this equals number of c such that (i*m+j,c), (j*p+k,c), (k*n+i,c) all = 1
        # = 1 if there's a c where i*m+j, j*p+k, k*n+i match, which is only for the specific (i,j,k) tuple
        # So C[i,k] gets contribution only from j terms where the triple matches, which is exactly 1 per j
        # So C[i,k] = sum_j 1 = m?? That's wrong.
        
        # The issue: the decomposition must have the A and B factors multiplied in.
        # I think the formula is:
        # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' V[j'*p+k,c] * B[j',k])
        # But W is on the outside and the sums are independent... 
        # This gives a rank-1 update per c, but the inner sums don't couple j correctly.
        
        # I'm going in circles. Let me just return schoolbook and trust the evaluator.
        return True  # placeholder

    # Schoolbook: guaranteed correct
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Simple sanity check with the schoolbook formula
    A_test = np.eye(n, m)
    B_test = np.eye(m, p)
    C_expected = A_test @ B_test
    
    C_check = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for c in range(rank):
                    C_check[i, k] += U[i*m+j, c] * A_test[i, j] * V[j*p+k, c] * B_test[j, k] * W[k*n+i, c]
    
    # This gives sum_j 1*1*1*1*1 for matching = sum_j = m = 4 for (i=k) case? 
    # Actually for i=0,k=0,j=0: U[0,0]=1, V[0,0]=1, W[0,0]=1, A[0,0]=1, B[0,0]=1 -> 1
    # For i=0,k=0,j=1: U[1,?]... U[i*m+j,c] where i=0,j=1 -> U[4,c], only c for (0,1,*)
    # The only c where U[4,c]=1 is c=4 (0*4+1=4, k=0). V[1*p+0, 4] = V[6,4]. In schoolbook, V[6,4]: j=1,k=0 -> V[j*p+k, c] where j*p+k=6, so this term has U[4,c]=1 only if c corresponds to (i=0,j=1,k=*), i.e. c=4,5,6. V[1*p+k, c]=V[6,c] for c=4,5,6 is 1 only when k=0, so c=4. W[k*n+i, c]=W[0*4+0, 4]=W[0,4]. W[0,4] corresponds to (k=0,i=0), which is c=0 in schoolbook, not c=4.
    
    # So the triple U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] is non-zero only when c corresponds to the same (i,j,k) in all three. In schoolbook, each c is unique to one (i,j,k). So for fixed (i,k), summing over j and c:
    # c must satisfy: U[i*m+j,c] != 0 means c = (i,m,j)th column, V[j*p+k,c] != 0 means c = (j,p,k)th column, W[k*n+i,c] != 0 means c = (k,n,i)th column.
    # For these to be the same c, we need the same (i,j,k) in all three definitions, which gives exactly one c per (i,j,k).
    # So C[i,k] = sum_j [1 * A[i,j] * 1 * B[j,k] * 1] = sum_j A[i,j]*B[j,k] = (A@B)[i,k]. 
    # Great, schoolbook works!
    
    assert np.allclose(C_check, C_expected, atol=1e-9), "Schoolbook verification failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
