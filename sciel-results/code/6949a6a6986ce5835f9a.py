"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This is the baseline; we try to find a lower-rank decomposition via search.
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    def schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, n, m, p, rank):
        # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # Actually, the matrix multiplication tensor is: T[i,j,k] = 1 if this is the (i,j) entry of A times (j,k) of B
        # The tensor for matrix multiplication C = AB where A is n x m, B is m x p, C is n x p
        # T[i, j, k] represents: C[i,k] += A[i,j] * B[j,k]
        # So the tensor entries are T[i, j, k] = 1 for all valid (i,j,k), and the decomposition is:
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, let me re-read the task. The output factor follows W[k*n+i, r].
        # So the tensor is indexed as: for matrix mult C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has entries T[i, j, k] which corresponds to the term A[i,j]*B[j,k] contributing to C[i,k]
        # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # But T[i,j,k] should be 1 for the "structural" tensor... 
        # Actually for matrix multiplication, the tensor is the "matrix multiplication tensor" which has
        # T[i, j, k] = 1 for all i in [n], j in [m], k in [p]. Wait no.
        # 
        # Let me think again. The matrix multiplication tensor mu_{n,m,p} has:
        # mu_{n,m,p}(i, j, k) = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
        # Actually the standard definition: the tensor T in (R^n*)^* \otimes (R^m*)^* \otimes (R^p*)^* 
        # has T(e_i^*, e_j^*, e_k^*) = 1 for the term A_ij * B_jk contributing to C_ik.
        # So T[i, j, k] = 1 for all valid indices? No...
        # 
        # Actually T[i, j, k] = 1 means: the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) of C.
        # So T[i, j, k] = 1 for ALL i, j, k where the indices are valid. There's no delta here.
        # The schoolbook decomposition uses rank n*m*p with each term picking out one specific (i,j,k).
        # 
        # So verify: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1.
        # But that's not right either, because different (i,j,k) tuples share the same U, V, W entries.
        # 
        # Wait, I think I'm confusing myself. Let me re-read:
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # The decomposition is: the matrix multiplication tensor T of shape (n*m, m*p, p*n)
        # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # The tensor T[a, b, c] = 1 if a encodes (i,j), b encodes (j,k), c encodes (k,i) with matching j and k.
        # Specifically: T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
        # 
        # So the verification is: for all i, j, k:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for all other combinations (i,j) in U, (j',k) in V, (k',i') in W where j != j' or k != k':
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] == 0
        
        # This is expensive to check fully, but we can spot-check
        import random
        # Check the "diagonal" entries should be 1
        for _ in range(200):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            s = 0.0
            for r in range(rank):
                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            if abs(s - 1.0) > 1e-6:
                return False
        # Check some off-diagonal entries should be 0
        for _ in range(200):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            kn = random.randint(0, p-1)
            ii = random.randint(0, n-1)
            if jp == j and kn == k:
                continue
            s = 0.0
            for r in range(rank):
                s += U[i*m+j][r] * V[jp*p+k][r] * W[kn*n+ii][r]
            if abs(s) > 1e-6:
                return False
        return True
    
    def verify_full(U, V, W, n, m, p, rank):
        # Full verification - expensive but necessary for correctness
        # Build the full tensor and check
        # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j==j' and k==k'
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kn in range(p):
                            for ii in range(n):
                                s = 0.0
                                for r in range(rank):
                                    s += U[i*m+j][r] * V[jp*p+k][r] * W[kn*n+ii][r]
                                expected = 1.0 if (jp == j and kn == k) else 0.0
                                if abs(s - expected) > 1e-8:
                                    return False
        return True
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try a few known lower-rank decompositions or search
    # For 3x5x6, schoolbook is 90. Let's try to find something better.
    # 
    # One approach: use the fact that we can combine some terms.
    # Another: use known results. The rank of mu_{3,5,6} is at most 90 (schoolbook).
    # 
    # Let's try a structured approach: partition the problem.
    # 
    # Actually, let's just return schoolbook for now since finding a lower rank
    # requires significant search time.
    
    # Try a simple improvement: if one dimension is 1, we can do better.
    # For general case, let's try random initialization + gradient-free search?
    # That's risky with 120s limit.
    
    # Let's try: rank = n*m*p but see if we can merge some columns.
    # For now, return schoolbook.
    return schoolbook()
# EVOLVE-BLOCK-END
