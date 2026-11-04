"""Matrix multiplication 3x4x4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _make_schoolbook(n, m, p):
    """Build the schoolbook (trivial) rank n*m*p decomposition."""
    total = n * m * p
    U = np.zeros((n * m, total))
    V = np.zeros((m * p, total))
    W = np.zeros((p * n, total))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2_decomposition():
    """
    Return (U, V, W) for the 2x2x2 matrix multiplication tensor
    using Strassen's 7-product algorithm.
    
    U has shape (4, 7) indexed by (i,j) -> row
    V has shape (4, 7) indexed by (j,k) -> row
    W has shape (4, 7) indexed by (k,i) -> row
    
    The identity is:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    where n=m=p=2 here.
    """
    # Strassen's 7 products for multiplying 2x2 matrices:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # For the tensor T[i,j,k] = 1 (n=m=p=2):
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Let's index: i in {0,1}, j in {0,1}, k in {0,1}
    # U row index: i*2+j
    # V row index: j*2+k
    # W row index: k*2+i
    
    # The 7 terms with their contributions to each (i,j,k):
    # Term 1: M1 = (a+d)(e+h)
    #   a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    #   This contributes to:
    #   C[0,0]: +1 (from a*e)
    #   C[0,1]: +1 (from a*h)
    #   C[1,0]: +1 (from d*e)
    #   C[1,1]: +1 (from d*h)
    
    # But in the tensor formulation, we're computing:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # And the tensor identity sum_c U[i,j,c]V[j,k,c]W[k,i,c] = 1
    # means that when we plug in:
    # sum_c W[k,i,c] (sum_j U[i,j,c] A[i,j]) (sum_j' V[j',k,c] B[j',k])
    # we get the correct C[i,k].
    
    # So for Strassen, each product M_c corresponds to:
    # (sum_{i,j} U[i,j,c] A[i,j]) * (sum_{j,k} V[j,k,c] B[j,k]) * W[k,i,c]
    
    # Let me re-derive. For M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]):
    # U part: U[0,0,1]=1, U[1,1,1]=1, others 0 for c=1
    # V part: V[0,0,1]=1, V[1,1,1]=1, others 0 for c=1
    # W part: We need this to contribute correctly to C[0,0], C[0,1], C[1,0], C[1,1]
    # The product (A[0,0]+A[1,1])(B[0,0]+B[1,1]) = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # This should contribute:
    #   to C[0,0]: A[0,0]B[0,0] -> coefficient +1
    #   to C[0,1]: A[0,0]B[1,1] -> coefficient +1  
    #   to C[1,0]: A[1,1]B[0,0] -> coefficient +1
    #   to C[1,1]: A[1,1]B[1,1] -> coefficient +1
    # So W[0,0,1]=1, W[1,0,1]=1, W[0,1,1]=1, W[1,1,1]=1
    
    # Let me just set up the 7 terms directly:
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A[0,0], A[1,1] (row indices: 0*2+0=0, 1*2+1=3)
    V[0, 0] = 1; V[3, 0] = 1  # B[0,0], B[1,1] (row indices: 0*2+0=0, 1*2+1=3)
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1  # all (k,i) combos
    
    # Term 2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A[1,0], A[1,1] (row indices: 1*2+0=2, 1*2+1=3)
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1; W[3, 1] = 1  # contributes to C[1,0] and C[1,1]... 
    # Wait, let me be more careful.
    # (A10+A11)B00 = A10*B00 + A11*B00
    # A10*B00 contributes to C[1,0] (since j=0, k=0, i=1)
    # A11*B00 contributes to C[1,0] (since j=1, k=0, i=1)
    # So only C[1,0] gets contribution.
    # W[0*2+1, 1] = W[1,1] = 1, others 0
    W[2, 1] = 0; W[3, 1] = 0
    W[1, 1] = 1  # k=0, i=1 -> index 0*2+1=1
    
    # Term 3: A00*(B01-B11)
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1; V[3, 2] = -1  # B[0,1], B[1,1]
    # A00*B01 contributes to C[0,1] (i=0,j=0,k=1)
    # -A00*B11 contributes to -C[0,1]... wait no
    # A00*(B01-B11) = A00*B01 - A00*B11
    # A00*B01: i=0,j=0,k=1 -> C[0,1]
    # A00*B11: i=0,j=1,k=1 -> C[0,1]
    # So both contribute to C[0,1]
    W[3, 2] = 1  # k=1, i=0 -> index 1*2+0=3
    
    # Term 4: A11*(B10-B00)
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1; V[0, 3] = -1  # B[1,0], B[0,0]
    # A11*B10: i=1,j=1,k=0 -> C[1,0]
    # -A11*B00: i=1,j=0,k=0 -> C[1,0]
    W[1, 3] = 1  # k=0, i=1 -> index 0*2+1=1
    
    # Term 5: (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1  # A[0,0], A[0,1]
    V[3, 4] = 1  # B[1,1]
    # A00*B11: i=0,j=0,k=1 -> C[0,1]
    # A01*B11: i=0,j=1,k=1 -> C[0,1]
    W[3, 4] = 1  # k=1, i=0 -> index 1*2+0=3
    
    # Term 6: (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A[1,0], A[0,0]
    V[0, 5] = 1; V[1, 5] = 1  # B[0,0], B[0,1]
    # A10*B00: i=1,j=0,k=0 -> C[1,0]
    # A10*B01: i=1,j=0,k=1 -> C[1,1]
    # -A00*B00: i=0,j=0,k=0 -> C[0,0]
    # -A00*B01: i=0,j=0,k=1 -> C[0,1]
    W[1, 5] = 1  # k=0, i=1
    W[3, 5] = 1  # k=1, i=1
    W[0, 5] = -1  # k=0, i=0
    W[2, 5] = -1  # k=1, i=0
    
    # Term 7: (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A[0,1], A[1,1]
    V[2, 6] = 1; V[3, 6] = 1  # B[1,0], B[1,1]
    # A01*B10: i=0,j=1,k=0 -> C[0,0]
    # A01*B11: i=0,j=1,k=1 -> C[0,1]
    # -A11*B10: i=1,j=1,k=0 -> C[1,0]
    # -A11*B11: i=1,j=1,k=1 -> C[1,1]
    W[0, 6] = 1  # k=0, i=0
    W[2, 6] = 1  # k=1, i=0
    W[1, 6] = -1  # k=0, i=1
    W[3, 6] = -1  # k=1, i=1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # First try the Strassen-based structured approach
    # Strategy: Use Strassen's 7-product algorithm on overlapping 2x2 blocks
    # For 3x4x4, we can try to exploit structure.
    
    # Approach: Build a rank-29 decomposition using:
    # - For rows i=0,1 (top 2x4 part): use Strassen-like structure
    # - For row i=2 (bottom 1x4 part): use rank-1
    
    # Actually, let me think about this more carefully.
    # The tensor is T[i,j,k] = 1 for i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
    
    # Simple approach: just use schoolbook and return it (rank 48)
    # But let's try to do better.
    
    # Try 1: Direct rank-47, 46, ... search is too slow.
    # Try 2: Use the Kronecker product structure.
    
    # Let's try a block approach:
    # Split the 3x4x4 into overlapping 2x2x2 blocks.
    
    # Actually, the simplest improvement: use Strassen for each 2x2x2 sub-block
    # But the dimensions don't align cleanly.
    
    # Let me just try to construct a valid lower-rank decomposition using 
    # the known fact that matmul(2,2,2) has rank 7.
    
    # For 3x4x4: 
    # We can view it as combining:
    # - A 2x4x4 block (rows 0,1)
    # - A 1x4x4 block (row 2)
    
    # For the 2x4x4 block, if we can get rank 28 (7*4), and for 1x4x4 rank 1,
    # total rank = 29.
    
    # Let's construct the 2x4x4 decomposition:
    # For the top 2 rows (i=0,1), we have a 2x4x4 matrix multiplication.
    # We can use Strassen's algorithm by viewing the 4x4 B matrix as 2x2 blocks
    # and the 2x4 A as 2x2 blocks.
    
    # Actually, let me use a different approach. 
    # The key insight: for matmul(n,m,p), if we have a rank-r decomposition of matmul(2,2,2),
    # we can use it to build decompositions of larger matmul tensors.
    
    # For 3x4x4, let's try:
    # Partition the j index (size 4) into two groups of 2.
    # Then A is 3x4 = [A1 | A2] where A1, A2 are 3x2
    # B is 4x4 = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = A1*B_top + A2*B_bottom where B_top = [B11 B12] (2x4) and B_bottom = [B21 B22] (2x4)
    
    # This doesn't directly help with Strassen.
    
    # Let me just return the schoolbook decomposition as a safe fallback,
    # but first try a few structured approaches.
    
    # Approach: Try to use the 7-term Strassen decomposition for 2x2x2
    # and extend it to 3x4x4.
    
    # For now, let's construct a rank-48 schoolbook decomposition and verify it,
    # then try to reduce.
    
    U, V, W = _make_schoolbook(n, m, p)
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook verification failed"
    
    # Now try to find a lower-rank decomposition using structured methods
    # Try rank 47 down to 30 using random initialization + ALS
    
    for target_rank in range(47, 30, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random initializations
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            U_try = np.random.randn(n * m, target_rank)
            V_try = np.random.randn(m * p, target_rank)
            W_try = np.random.randn(p * n, target_rank)
            
            # ALS iterations
            for iteration in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                for idx in range(n * m):
                    i, j = idx // m, idx % m
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        for c in range(target_rank):
                            A_mat[k, c] = V_try[j * p + k, c] * W_try[k * n + i, c]
                    b_vec = np.ones(p)
                    try:
                        U_try[idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for idx in range(m * p):
                    j, k = idx // p, idx % p
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        for c in range(target_rank):
                            A_mat[i, c] = U_try[i * m + j, c] * W_try[k * n + i, c]
                    b_vec = np.ones(n)
                    try:
                        V_try[idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for idx in range(p * n):
                    k, i = idx // n, idx % n
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            A_mat[j, c] = U_try[i * m + j, c] * V_try[j * p + k, c]
                    b_vec = np.ones(m)
                    try:
                        W_try[idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    U, V, W = U_try, V_try, W_try
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the best we found (schoolbook)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    valid = _verify_decomposition(U, V, W, n, m, p)
    print(f"Rank: {rank}, Valid: {valid}")
