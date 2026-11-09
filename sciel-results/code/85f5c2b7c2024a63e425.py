"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Recursive Kronecker-product construction.
    # We build a decomposition for (n,m,p) by splitting along the m-dimension.
    # The base case (2,2,2) uses Strassen's rank-7 algorithm.
    #
    # The decomposition tensor T is a 3-way tensor where T[i,j,k] = 1 if
    # the product C[i,k] depends on A[i,j]*B[j,k].
    #
    # In the factorization U V W, we have:
    #   T[i,j,k] = sum_r U[i,j,r] * V[r] * W[k,i,r]
    # where V[r] is a scalar (or we can absorb it into U or W).
    # Actually, the standard form is:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Let's define the recursion more carefully.
    #
    # For matrix multiplication of A (n x m) and B (m x p), giving C (n x p):
    # T is a tensor of shape (n, m, p) where T[i,j,k] = 1.
    #
    # We can split the middle dimension m = m1 + m2:
    # A = [A1 | A2] where A1 is n x m1, A2 is n x m2
    # B = [B1; B2] where B1 is m1 x p, B2 is m2 x p
    # C = A1*B1 + A2*B2
    #
    # So T(n,m,p) = T(n,m1,p) + T(n,m2,p) with appropriate indexing.
    #
    # The rank of T(n,m,p) <= rank(T(n,m1,p)) + rank(T(n,m2,p)).
    #
    # For the base case (2,2,2), we use Strassen's algorithm which gives rank 7.
    #
    # Let's verify the factorization convention:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strassen decomposition for 2x2x2
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let's write this in terms of U, V, W.
    # For each term, we need U[i,j,r], V[j,k,r], W[k,i,r] such that
    # the product sums to give the right C elements.
    
    # Actually, let me think about this differently.
    # The tensor T for (n,m,p) matrix multiplication has T[i,j,k] = 1 for all i,j,k.
    # We want to find rank-r decomposition: T = sum_r u_r * v_r * w_r (outer product)
    # where u_r is a vector of length n*m, v_r is length m*p, w_r is length p*n.
    #
    # T[i,j,k] = sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
    
    # Strassen for (2,2,2):
    # n=2, m=2, p=2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    
    # Let's define the 7 Strassen terms.
    # Term 1: (A00+A11)(B00+B11)
    # This contributes to C00, C11 with +1, and we'll handle cross terms separately.
    
    # Actually, the standard Strassen decomposition as a tensor decomposition:
    # Each of the 7 multiplications gives a rank-1 term.
    
    # Let me look up the explicit tensor decomposition for Strassen.
    # The 7 rank-1 tensors are:
    # 
    # For multiplication M1 = (A00+A11)(B00+B11):
    # u1: [1, 0, 1, 0] (for A00, A01, A10, A11)
    # v1: [1, 0, 0, 1] (for B00, B01, B10, B11)
    # w1: [1, 0, 0, 1] (for C00, C01, C10, C11) - but wait, C00 and C11 both get +M1
    
    # Hmm, this is getting complex. Let me use a different approach.
    #
    # I'll directly construct the decomposition using the recursive splitting
    # and verify it numerically.
    
    def build_decomposition(n, m, p):
        """Build a decomposition for n x m x p matrix multiplication.
        
        Returns U (n*m x rank), V (m*p x rank), W (p*n x rank)
        such that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        # Base case: 2x2x2, use Strassen
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        # Recursive case: split m into m1 + m2
        # Choose split: try to split into even parts if possible
        m1 = m // 2
        m2 = m - m1
        
        # Get decompositions for the two halves
        U1, V1, W1 = build_decomposition(n, m1, p)
        U2, V2, W2 = build_decomposition(n, m2, p)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 + r2
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # First part: columns 0..m1-1 of A, rows 0..m1-1 of B
        # U1 has shape (n*m1, r1). We need to map to (n*m, r1)
        # For i in 0..n-1, j in 0..m1-1: U[i*m+j, :] = U1[i*m1+j, :]
        for i in range(n):
            for j in range(m1):
                U[i*m+j, :r1] = U1[i*m1+j, :]
        
        # V1 has shape (m1*p, r1). Map to (m*p, r1)
        # For j in 0..m1-1, k in 0..p-1: V[j*p+k, :] = V1[j*p+k, :]
        for j in range(m1):
            for k in range(p):
                V[j*p+k, :r1] = V1[j*p+k, :]
        
        # W1 has shape (p*n, r1). Same indexing.
        W[:, :r1] = W1
        
        # Second part: columns m1..m-1 of A, rows m1..m-1 of B
        # U2 has shape (n*m2, r2). Map to (n*m, r2)
        # For i in 0..n-1, j in m1..m-1: U[i*m+j, r1:] = U2[i*m2+(j-m1), r1-r1:]
        for i in range(n):
            for j in range(m1, m):
                U[i*m+j, r1:] = U2[i*m2+(j-m1), :]
        
        # V2 has shape (m2*p, r2). Map to (m*p, r2)
        # For j in m1..m-1, k in 0..p-1: V[j*p+k, r1:] = V2[(j-m1)*p+k, :]
        for j in range(m1, m):
            for k in range(p):
                V[j*p+k, r1:] = V2[(j-m1)*p+k, :]
        
        # W2 has shape (p*n, r2)
        W[:, r1:] = W2
        
        return U, V, W
    
    def strassen_2x2x2():
        """Strassen decomposition for 2x2x2.
        
        Returns U (4x7), V (4x7), W (4x7)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B00=0, B01=1, B10=2, B11=3
        # C indices: C00=0, C01=1, C10=2, C11=3
        
        # M1: (A00+A11)(B00+B11)
        U[:, 0] = [1, 0, 0, 1]  # A00 + A11
        V[:, 0] = [1, 0, 0, 1]  # B00 + B11
        W[:, 0] = [1, 0, 0, 1]  # C00 + C11
        
        # M2: (A10+A11)B00
        U[:, 1] = [0, 0, 1, 1]  # A10 + A11
        V[:, 1] = [1, 0, 0, 0]  # B00
        W[:, 1] = [0, 0, 1, -1]  # C10 - C11
        
        # M3: A00(B10-B11)
        U[:, 2] = [1, 0, 0, 0]  # A00
        V[:, 2] = [0, 0, 1, -1]  # B10 - B11
        W[:, 2] = [0, 1, 0, 1]  # C01 + C11
        
        # M4: A11(B01-B10)
        U[:, 3] = [0, 0, 0, 1]  # A11
        V[:, 3] = [0, 1, -1, 0]  # B01 - B10
        W[:, 3] = [1, 0, 1, 0]  # C00 + C10
        
        # M5: (A00+A01)B11
        U[:, 4] = [1, 1, 0, 0]  # A00 + A01
        V[:, 4] = [0, 0, 0, 1]  # B11
        W[:, 4] = [-1, 1, 0, 0]  # -C00 + C01
        
        # M6: (A10-A00)(B00+B10)
        U[:, 5] = [-1, 0, 1, 0]  # -A00 + A10
        V[:, 5] = [1, 0, 1, 0]  # B00 + B10
        W[:, 5] = [0, 0, 0, 1]  # C11
        
        # M7: (A01-A11)(B01+B11)
        U[:, 6] = [0, 1, 0, -1]  # A01 - A11
        V[:, 6] = [0, 1, 0, 1]  # B01 + B11
        W[:, 6] = [1, 0, 0, 0]  # C00
        
        return U, V, W
    
    # Build the decomposition
    U, V, W = build_decomposition(n, m, p)
    
    rank = U.shape[1]
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all i,j,k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
