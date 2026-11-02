"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    The key insight from the measured knowledge: using two copies of a Rank 11 decomposition 
    for 2x3x3 gives a Rank 22 decomposition for 2x3x6.
    
    The matrix multiplication tensor T[i,j,k] = 1 if the indices correspond to a valid 
    multiplication term. For (n,m,p) = (2,3,6), the tensor has dimensions (n*m) x (m*p) x (p*n) = 6 x 18 x 12.
    
    We need to find rank-r matrices U (6 x r), V (18 x r), W (12 x r) such that:
    T = sum_{c=1}^{r} U[:,c] * V[:,c].T * W[:,c].T  (outer product structure)
    
    Specifically: T[a,b,c] = sum_{d} U[a,d] * V[b,d] * W[c,d]
    where a = i*m+j, b = j*p+k, c = k*n+i
    
    Strategy: Use the known result that 2x3x3 has rank 11, so 2x3x6 has rank 22.
    We construct this by splitting the 6 columns into two groups of 3.
    """
    n, m, p = DIMENSIONS
    # For 2x3x6, we can think of it as two 2x3x3 blocks
    # The rank should be 2 * 11 = 22
    
    # Let's construct the schoolbook decomposition first (rank = n*m*p = 36)
    # Then try to find a lower rank decomposition
    
    # Actually, let's directly construct the known rank-22 decomposition
    # by leveraging the 2x3x3 rank-11 decomposition
    
    # For 2x3x3, the multiplication tensor T_{233}[i,j,k] where:
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # T[i,j,k] = 1 if the (i,j) entry of first matrix and (j,k) entry of second matrix
    # contribute to the (i,k) entry of product
    
    # The standard schoolbook for 2x3x3 has rank 6
    # But there's a known rank-11 decomposition? Wait, that doesn't sound right.
    # Let me reconsider.
    
    # Actually, for matrix multiplication (n,m,p), the schoolbook rank is n*m*p.
    # For 2x3x3, that's 18. For 2x3x6, that's 36.
    
    # The measured knowledge says "Rank 11 decomposition for 2x3x3" which seems unusual.
    # Let me think about this differently.
    
    # Perhaps the decomposition exploits structure. Let me try to find a lower-rank 
    # decomposition using the fact that some entries can be shared.
    
    # Let me try a direct approach: construct the tensor and try to find its rank
    # via random initialization and iterative refinement, or use a known algebraic decomposition.
    
    # For now, let's use the schoolbook (rank 36) as a safe baseline, and then
    # attempt to reduce the rank.
    
    # Actually, let me think more carefully. The tensor for (2,3,6) multiplication:
    # A is 2x3, B is 3x6, C = A*B is 2x6
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # The tensor T has indices (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,1,5}
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # This is just a 2x3x6 all-ones tensor in the relevant subspace.
    # Wait, no. The tensor is defined by the multiplication structure.
    
    # Let me re-read the problem. The output factor follows W[k*n+i, r].
    # So W is indexed by (k, i) where k is the column of B / row of C, and i is the row of A / C.
    
    # The tensor T[a, b, c] where:
    # a = i*m + j (row of U, corresponding to A[i,j])
    # b = j*p + k (row of V, corresponding to B[j,k])  
    # c = k*n + i (row of W, corresponding to C[i,k])
    # T[a, b, c] = 1 if a, b, c are consistent (i.e., they share the same i, j, k)
    
    # For the schoolbook, each (i,j,k) triple gives one rank-1 term.
    
    # Can we do better? The key is that different (i,j,k) triples might share structure.
    
    # Let me try to find a rank-22 decomposition by exploiting the 2x3x3 block structure.
    # Split k into k1 in {0,1,2} and k2 in {3,4,5}, so B = [B1 | B2] where B1, B2 are 3x3.
    # Then C = A * [B1 | B2] = [A*B1 | A*B2]
    
    # So the 2x3x6 multiplication is two independent 2x3x3 multiplications.
    # If 2x3x3 has rank r, then 2x3x6 has rank 2r.
    
    # What's the rank of 2x3x3 matrix multiplication?
    # The schoolbook rank is 2*3*3 = 18.
    # It's known that the rank of matrix multiplication can be less than schoolbook.
    # For 2x2x2, the rank is 7 (Strassen).
    # For 2x3x3, I'm not sure of the exact rank.
    
    # Given the measured knowledge says rank 11 for 2x3x3, let's trust that and construct rank 22.
    
    # Let me try to find the rank-11 decomposition for 2x3x3 by solving a system of equations.
    
    def find_233_decomposition():
        """Find a low-rank decomposition for 2x3x3 matrix multiplication."""
        n2, m2, p2 = 2, 3, 3
        # Tensor dimensions: (n2*m2) x (m2*p2) x (p2*n2) = 6 x 9 x 6
        # T[a,b,c] = 1 if a=i*m2+j, b=j*p2+k, c=k*n2+i for some i,j,k
        
        # Create the tensor
        T = np.zeros((n2*m2, m2*p2, p2*n2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a = i*m2 + j
                    b = j*p2 + k
                    c = k*n2 + i
                    T[a, b, c] = 1
        
        # Try to find a rank-r decomposition
        # U (6 x r), V (9 x r), W (6 x r) such that T = sum_d U[:,d] * V[:,d].T * W[:,d].T
        
        # This is equivalent to: for each (a,b,c), sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
        
        # Let's try r = 11
        r = 11
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n2*m2, r)
        V = np.random.randn(m2*p2, r)
        W = np.random.randn(p2*n2, r)
        
        # Use iterative refinement: for each d, optimize U[:,d], V[:,d], W[:,d]
        # to minimize the error
        
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        for trial in range(20):
            U = np.random.randn(n2*m2, r)
            V = np.random.randn(m2*p2, r)
            W = np.random.randn(p2*n2, r)
            
            # Iterate
            for it in range(200):
                # Update U
                for d in range(r):
                    # Fix V[:,d] and W[:,d], optimize U[:,d]
                    # T[a,b,c] = sum_d' U[a,d']*V[b,d']*W[c,d']
                    # For fixed d, the contribution to T from term d is U[:,d] * V[:,d].T * W[:,d].T
                    # We want: sum_{d'} U[a,d']*V[b,d']*W[c,d'] = T[a,b,c]
                    # This is a linear system in U[:,d] if we fix other terms.
                    
                    # Let R = T - sum_{d' != d} U[:,d'] * V[:,d'].T * W[:,d'].T
                    # Then we need U[:,d] * V[:,d].T * W[:,d].T = R
                    # This is: for each (b,c), R[:,b,c] = U[:,d] * (V[b,d]*W[c,d])
                    # Let vwc = V[:,d] * W[:,d] (element-wise, shape 9x6)
                    # Then R = U[:,d] * vwc (outer product in a sense)
                    # R[a,b,c] = U[a,d] * vwc[b,c]
                    # This is a rank-1 approximation problem.
                    
                    # Actually, let me use a simpler approach: solve the linear system directly.
                    # Reshape T into a matrix and use SVD or similar.
                    
                    pass  # This is getting complex. Let me use a different approach.
            
            # Compute error
            err = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        a = i*m2 + j
                        b = j*p2 + k
                        c = k*n2 + i
                        val = 0
                        for d in range(r):
                            val += U[a,d] * V[b,d] * W[c,d]
                        err += (val - T[a,b,c])**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        return best_U, best_V, best_W, best_err
    
    # Actually, let me try a more direct algebraic approach.
    # For 2x3x3, let's denote the entries of A as a00, a01, a02, a10, a11, a12
    # Entries of B as b00, b01, b02, b10, b11, b12, b20, b21, b22
    # Entries of C as c00, c01, c02, c10, c11, c12
    
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # That's 6 output entries, each a sum of 3 products = 18 products total (schoolbook).
    
    # Can we compute these with fewer multiplications?
    # This is a specific instance of the matrix multiplication rank problem.
    
    # Given the complexity of finding an optimal decomposition analytically, and given
    # that the measured knowledge indicates rank 22 works, let me construct it directly
    # by finding a rank-11 decomposition for 2x3x3 numerically.
    
    # Let me use a different strategy: set up the equations and solve.
    
    def solve_233_rank(r):
        """Solve for rank-r decomposition of 2x3x3 multiplication tensor."""
        n2, m2, p2 = 2, 3, 3
        dim_u, dim_v, dim_w = n2*m2, m2*p2, p2*n2
        
        # Number of constraints: dim_u * dim_v * dim_w = 6*9*6 = 324
        # But many are zero. Non-zero entries: n2*m2*p2 = 18
        # Actually, the tensor has exactly 18 non-zero entries (all 1s at valid positions).
        
        # Unknowns: r * (dim_u + dim_v + dim_w) = r * (6 + 9 + 6) = r * 21
        # For r = 11: 231 unknowns, 18 equations (one per non-zero tensor entry)
        # Wait, that's not right. Each non-zero entry gives one equation, but the equations
        # are nonlinear (products of variables).
        
        # Let me set it up properly. For each (i,j,k):
        # sum_d U[i*m2+j, d] * V[j*p2+k, d] * W[k*n2+i, d] = 1
        
        # This is a system of 18 equations in 231 unknowns (for r=11).
        # We can solve this by fixing some variables and solving for others.
        
        # Let me fix W to be an identity-like matrix and solve for U and V.
        # Actually, let me use a numerical approach with scipy.
        
        from scipy.optimize import least_squares
        
        def residual(params):
            """Residual function for the rank-r decomposition."""
            # params: flattened [U (6*r), V (9*r), W (6*r)]
            U = params[0:dim_u*r].reshape(dim_u, r)
            V = params[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
            W = params[dim_u*r+dim_v*r:dim_u*r+dim_v*r+dim_w*r].reshape(dim_w, r)
            
            res = []
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        a = i*m2 + j
                        b = j*p2 + k
                        c = k*n2 + i
                        val = 0
                        for d in range(r):
                            val += U[a,d] * V[b,d] * W[c,d]
                        res.append(val - 1)
            
            return np.array(res)
        
        # Initial guess
        np.random.seed(42)
        x0 = np.random.randn(r*(dim_u+dim_v+dim_w))
        
        result = least_squares(residual, x0, method='lm', max_nfev=10000)
        
        if result.success or np.linalg.norm(result.fun) < 1e-6:
            U = result.x[0:dim_u*r].reshape(dim_u, r)
            V = result.x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
            W = result.x[dim_u*r+dim_v*r:dim_u*r+dim_v*r+dim_w*r].reshape(dim_w, r)
            return U, V, W, result.cost
        else:
            return None, None, None, result.cost
    
    # Try to find rank-11 decomposition for 2x3x3
    U233, V233, W233, err233 = solve_233_rank(11)
    
    if U233 is not None and err233 < 1e-8:
        # Construct 2x3x6 decomposition from two 2x3x3 decompositions
        n, m, p = DIMENSIONS
        p_half = 3  # Split 6 into two 3s
        
        # U is 6 x 22, V is 18 x 22, W is 12 x 22
        U = np.zeros((n*m, 22))
        V = np.zeros((m*p, 22))
        W = np.zeros((p*n, 22))
        
        # First half (k=0,1,2)
        for i in range(n):
            for j in range(m):
                U[i*m+j, 0:11] = U233[i*m+j, :]
        
        for j in range(m):
            for k in range(p_half):
                V[j*p+k, 0:11] = V233[j*p_half+k, :]
        
        for k in range(p_half):
            for i in range(n):
                W[k*n+i, 0:11] = W233[k*n+i, :]
        
        # Second half (k=3,4,5)
        for i in range(n):
            for j in range(m):
                U[i*m+j, 11:22] = U233[i*m+j, :]
        
        for j in range(m):
            for k in range(p_half, p):
                V[j*p+k, 11:22] = V233[j*p_half+(k-p_half), :]
        
        for k in range(p_half, p):
            for i in range(n):
                W[k*n+i, 11:22] = W233[(k-p_half)*n+i, :]
        
        # Verify
        # Check that the decomposition is correct
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = 0
                    for d in range(22):
                        val += U[a,d] * V[b,d] * W[c,d]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-6:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition (rank 36)
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
# EVOLVE-BLOCK-END
