import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A is (n, m) = (2, 3), B is (m, p) = (3, 6), C is (n, p) = (2, 6)
    
    # We aim to find a rank-minimal decomposition.
    # Schoolbook rank is n*m*p = 2*3*6 = 36.
    # We suspect a lower rank is possible.
    # Let's try to construct a decomposition of rank 33 using a block structure.
    # The matrix B is 3x6. We can split the 6 columns into two groups of 3.
    # Let B = [B1 | B2] where B1, B2 are 3x3.
    # Then C = A * B = A * [B1 | B2] = [A*B1 | A*B2].
    # So we need to compute two products of 2x3 by 3x3 matrices.
    # Let's try to find the minimal rank for 2x3 by 3x3.
    # Schoolbook rank for 2x3x3 is 18.
    # If we use rank 18 for each, total rank is 36.
    # Can we do better for 2x3x3?
    # Let's try to find a decomposition for 2x3x3 with rank < 18.
    # Or maybe we can mix the computations?
    
    # Let's try a different approach.
    # We will formulate the problem as finding a tensor decomposition of rank r.
    # The tensor T has indices (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # T[i, j, k] = 1 if the indices correspond to a valid multiplication term?
    # No, the matrix multiplication tensor is defined by T[i, j, k] = delta_{j, j'} for the middle index?
    # Wait, the tensor for matrix multiplication C = A * B is:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor T is of shape (n, m, p) such that T[i, j, k] = 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # Actually, the tensor is defined such that the bilinear map (A, B) -> C is given by:
    # C = sum_{r=1}^R U_r(A) V_r(B) W_r.
    # Where U_r(A) is a scalar, V_r(B) is a scalar, and W_r is a matrix.
    # Or more standardly: C_{i, k} = sum_r U_{i, j, r} A_{i, j} V_{j, k, r} B_{j, k}? No.
    # The standard decomposition is:
    # C_{i, k} = sum_r (sum_j U_{r, i, j} A_{i, j}) (sum_k V_{r, j, k} B_{j, k})? No.
    # Let's use the definition from the task:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # The decomposition is:
    # sum_{r} U[:, r] (x) V[:, r] (y) W[:, r] = T
    # where x is the vector of A entries, y is the vector of B entries.
    # The product U[:, r] (x) is a scalar: sum_{i,j} U[i*m+j, r] A[i,j].
    # The product V[:, r] (y) is a scalar: sum_{j,k} V[j*p+k, r] B[j,k].
    # The term W[:, r] is added to the tensor C.
    # W has indices (k*n+i, r). So W[k*n+i, r] is the coefficient for C[i,k].
    # So the equation is:
    # For all i, k:
    # C[i,k] = sum_r (sum_{j} U[i*m+j, r] A[i,j]) (sum_{j'} V[j'*p+k, r] B[j',k])?
    # No, the scalar products are separate.
    # Let S_r = sum_{i,j} U[i*m+j, r] A[i,j].
    # Let T_r = sum_{j,k} V[j*p+k, r] B[j,k].
    # Then C[i,k] = sum_r S_r T_r W[k*n+i, r].
    # This must equal sum_j A[i,j] B[j,k].
    
    # This is a system of equations for the entries of U, V, W.
    # We want to find the minimal rank r.
    
    # Let's try to solve for r = 33, 34, 35.
    # We can use a numerical solver or a heuristic search.
    # Given the constraints, a full Groebner basis is too slow.
    # We will try to construct a specific decomposition.
    
    # Hypothesis: Rank 33 is possible.
    # Let's try to verify if rank 33 works by solving a linear system?
    # No, it's bilinear.
    
    # Let's try a simpler block decomposition.
    # Split B into 3 blocks of 3x2? No, 3x6 -> three 3x2 blocks.
    # C = A * [B1 | B2 | B3] = [A*B1 | A*B2 | A*B3].
    # Each block is 2x2.
    # Rank of 2x2x2 is 7 (Strassen).
    # Total rank 3 * 7 = 21?
    # Wait, A is 2x3. B1 is 3x2.
    # The product is 2x3 by 3x2.
    # This is not a square matrix multiplication.
    # Can we embed 2x3x2 into a larger square multiplication?
    # Or use a specific algorithm for rectangular multiplication.
    
    # Let's check the rank of 2x3x2.
    # Schoolbook: 12.
    # Can we do better?
    # Let's try to find a decomposition of rank 11 or 10 for 2x3x2.
    # If we can do rank 11 for 2x3x2, then 3 * 11 = 33.
    
    # Let's attempt to construct a rank 11 decomposition for 2x3x2.
    # A is 2x3, B is 3x2, C is 2x2.
    # C = A * B.
    # Let's try to use a Groebner basis or numerical search for 2x3x2.
    # Since 2x3x2 is small, we can try to solve for rank 11.
    
    # We will implement a function to find a decomposition for 2x3x2 with rank r.
    # If found, we use it 3 times.
    
    def find_decomposition_232(rank):
        # n=2, m=3, p=2
        n, m, p = 2, 3, 2
        # Variables: U (6, r), V (6, r), W (4, r)
        # Total variables: 16 * r
        # Equations: 4 (for C entries) * (number of monomials in A,B)?
        # The identity must hold for all A, B.
        # This means the tensor T must be decomposed.
        # T is a tensor of shape (2, 3, 2) with T[i,j,k] = 1?
        # No, the tensor for multiplication is not just 1s.
        # The condition is that the map (A,B) -> C is linear in A and B.
        # The tensor T_{i,j,k} is defined such that C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
        # In the decomposition:
        # C_{i,k} = sum_r (sum_{j} U_{i,j,r} A_{i,j}) (sum_{j'} V_{j',k,r} B_{j',k}) W_{i,k,r}?
        # Wait, the indices in the task description are:
        # U (n*m, rank) -> U[i*m+j, r]
        # V (m*p, rank) -> V[j*p+k, r]
        # W (p*n, rank) -> W[k*n+i, r]
        
        # The term is:
        # sum_r U[i*m+j, r] A[i,j] * V[j'*p+k, r] B[j',k] * W[k*n+i, r]
        # This doesn't look right. The sums over j and j' are independent in the scalar products?
        # Let's re-read the standard definition.
        # Usually: C_{i,k} = sum_r U_{r,i,j} A_{i,j} V_{r,j,k} B_{j,k}.
        # Here the middle index j is shared.
        # In the task's interface:
        # U is flattened (i, j). V is flattened (j, k).
        # If we take the scalar product U_r . A, it sums over ALL i, j.
        # S_r = sum_{i,j} U[i*m+j, r] A[i,j].
        # T_r = sum_{j,k} V[j*p+k, r] B[j,k].
        # Then C[i,k] = sum_r S_r T_r W[k*n+i, r].
        # This implies that the decomposition does NOT depend on the specific i, j, k in the factors,
        # but rather the factors are global scalars.
        # This is the "rank" of the tensor in the sense of tensor rank.
        # For matrix multiplication, the tensor is T_{i,j,k} = delta_{j, j'}?
        # Actually, the tensor for matrix multiplication A(n,m) x B(m,p) -> C(n,p)
        # is defined by T[i,j,k] = 1 if the indices match the multiplication structure?
        # No, the tensor is simply the linear map.
        # The condition is:
        # For all i, k:
        # sum_{j} A[i,j] B[j,k] = sum_r (sum_{i',j'} U[i'*m+j', r] A[i',j']) (sum_{j'',k''} V[j''*p+k'', r] B[j'',k'']) W[k*n+i, r]
        
        # This is a very strong condition.
        # Let's check if this is the correct interpretation.
        # If so, the rank is the rank of the tensor T where T_{i,j,k} is the coefficient of A[i,j]B[j,k] in C[i,k]?
        # No, the RHS is a sum of products of global sums.
        # This means the decomposition is of the form:
        # C = sum_r (u_r^T A) (v_r^T B) w_r
        # where u_r is a vector of size n*m, v_r is a vector of size m*p, and w_r is a matrix of size n*p (flattened).
        # This is exactly the definition of tensor rank for the multiplication tensor.
        
        # So we need to find the minimal r such that there exist u_r, v_r, w_r satisfying the identity.
        
        # For 2x3x2:
        # n=2, m=3, p=2.
        # U (6, r), V (6, r), W (4, r).
        # We want to find if r=11 is possible.
        
        # We can set up a system of equations.
        # The identity must hold for all A, B.
        # This means the coefficient of each monomial A[i,j] B[j',k] must match.
        # LHS: Coeff of A[i,j] B[j',k] is 1 if j=j', else 0.
        # RHS: sum_r U[i'*m+j', r] V[j''*p+k'', r] W[k*n+i, r] * (delta_{i,i'} delta_{j,j'} delta_{j'',j'} delta_{k,k''}?)
        # Wait, the expansion of (sum u A) (sum v B) is sum_{i',j',j'',k''} U[i',j'] V[j'',k''] A[i',j'] B[j'',k''].
        # The coefficient of A[i,j] B[j',k] in the product is sum_r U[i,j,r] V[j',k,r].
        # Then we multiply by W[k*n+i, r] and sum over r.
        # So the coefficient of A[i,j] B[j',k] in C[i,k] is:
        # sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r].
        
        # We require:
        # If j == j': sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        # If j != j': sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = 0
        
        # This is a system of bilinear equations.
        # We can fix W and solve for U, V? Or fix U, V and solve for W?
        # If we fix U and V, the equations are linear in W.
        # For each (i, k, j, j'), we have an equation for W.
        # Actually, for each (i, k), the equation is:
        # sum_r (sum_{j} U[i*m+j, r] delta_{j,j'} V[j'*p+k, r]) W[k*n+i, r] = delta_{j,j'}
        # This is not quite right.
        # The equation is for each monomial A[i,j] B[j',k].
        # There are n*m * m*p = 6 * 6 = 36 monomials.
        # But they only contribute to specific C[i,k].
        # For a fixed (i, k), the monomials A[i,j] B[j',k] contribute to C[i,k].
        # The coefficient in LHS is 1 if j=j', else 0.
        # The coefficient in RHS is sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r].
        
        # So for each i, k, and each j, j':
        # sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = 1 if j==j' else 0.
        
        # This is a system of n*p*m*m = 2*2*3*3 = 36 equations.
        # Unknowns: r * (6 + 6 + 4) = 16r.
        # If r=11, 176 unknowns, 36 equations. Underdetermined.
        # We can try to find a solution.
        
        # Let's try to set W to be identity-like or simple.
        # If we set W[k*n+i, r] = delta_{r, idx(i,k)}, i.e., each W is a basis vector.
        # Then the equation becomes:
        # U[i*m+j, idx(i,k)] V[j'*p+k, idx(i,k)] = 1 if j==j' else 0.
        # This requires that for each (i,k), the vectors U[:, idx(i,k)] and V[:, idx(i,k)] are such that
        # their outer product (with indices i,j and j',k) is the identity matrix in j, j'.
        # But U[i*m+j, r] depends on i, and V[j'*p+k, r] depends on k.
        # This seems impossible if we use distinct r for each (i,k) because U and V would need to be zero elsewhere.
        # But we can reuse r.
        
        # Let's try a numerical approach.
        # We will use a random initialization and gradient descent?
        # Or we can use the fact that the equations are bilinear.
        # We can fix W to random values and solve for U, V?
        # The equations are linear in U if V and W are fixed?
        # sum_r U[i*m+j, r] (V[j'*p+k, r] W[k*n+i, r]) = delta_{j,j'}.
        # Let C_{j,j',i,k,r} = V[j'*p+k, r] W[k*n+i, r].
        # Then sum_r U[i*m+j, r] C_{j,j',i,k,r} = delta_{j,j'}.
        # This is linear in U.
        # We can set up a linear system for U given V, W.
        # There are n*m*r = 6r unknowns in U.
        # There are n*p*m*m = 36 equations.
        # If 6r >= 36, i.e., r >= 6, we might have a solution.
        # But the equations are coupled across i, k.
        
        # Let's try r=11.
        # We will iterate:
        # 1. Initialize V, W randomly.
        # 2. Solve for U from the linear system.
        # 3. Initialize U, W randomly.
        # 4. Solve for V from the linear system.
        # 5. Check residual.
        
        # This is an alternating least squares (ALS) method.
        
        import random
        random.seed(42)
        
        U = np.zeros((6, rank))
        V = np.zeros((6, rank))
        W = np.zeros((4, rank))
        
        # Initialize
        for _ in range(rank):
            V[:, _] = np.random.randn(6)
            W[:, _] = np.random.randn(4)
            
        for iteration in range(1000):
            # Solve for U
            # Equations: for each i, k, j, j':
            # sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j,j'}
            # Group by i, k.
            # For fixed i, k, the equations are:
            # For each j, j':
            # sum_r U[i*m+j, r] (V[j'*p+k, r] W[k*n+i, r]) = delta_{j,j'}
            # Let A_{j, r} = U[i*m+j, r].
            # Let B_{j', r} = V[j'*p+k, r] W[k*n+i, r].
            # We want sum_r A_{j, r} B_{j', r} = delta_{j,j'}.
            # This is A * B^T = I.
            # A is (m, r) = (3, r). B is (m, r) = (3, r).
            # We need to find A such that A * B^T = I.
            # This is possible if B has full rank (3).
            # If B is 3xr with r>=3, and rank 3, we can solve for A.
            # A = I * (B^T)^+ = (B^T)^+
            # So U[i*m+j, r] = ((V[:, k] * W[i])^T)^+ ?
            # Let's be precise.
            # For fixed i, k:
            # B_matrix is 3 x r. B_matrix[j', r] = V[j'*p+k, r] W[k*n+i, r].
            # We need A_matrix (3 x r) such that A_matrix * B_matrix.T = I_3.
            # A_matrix = I_3 * (B_matrix.T)^+ = (B_matrix.T)^+
            # So U[i*m+j, r] = ((B_matrix.T)^+)[j, r]
            
            B_mat = np.zeros((m, rank))
            for jp in range(m):
                B_mat[jp, :] = V[jp*p+k, :] * W[k*n+i, :]
            
            # Pseudo inverse of B_mat.T
            # B_mat.T is r x 3.
            # We want A (3 x r) such that A @ B_mat.T = I.
            # A = pinv(B_mat.T)
            try:
                A_mat = np.linalg.pinv(B_mat.T)
            except:
                continue
            
            for j in range(m):
                U[i*m+j, :] = A_mat[j, :]
            
            # Solve for V similarly
            # For fixed i, k:
            # We need B_matrix such that A_matrix * B_matrix.T = I.
            # B_matrix.T = A_matrix.T ^+
            # B_matrix = (A_matrix.T ^+)^T = pinv(A_matrix.T).T = pinv(A_matrix)
            # Wait, (X^+)^T = (X^T)^+.
            # So B_matrix = pinv(A_matrix).
            
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i*m+j, :]
            
            try:
                B_mat = np.linalg.pinv(A_mat)
            except:
                continue
            
            # B_mat[j', r] = V[j'*p+k, r] W[k*n+i, r]
            # We have B_mat from pinv.
            # But W is fixed.
            # V[j'*p+k, r] = B_mat[j', r] / W[k*n+i, r]
            # This requires W[k*n+i, r] != 0.
            # If W is zero, we have a problem.
            # We can update W as well.
            
            # Let's update W too.
            # This is getting complicated.
            # Let's just return the schoolbook rank for now if this doesn't work quickly.
            
            # Check residual
            residual = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = 0
                            for r in range(rank):
                                val += U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
                            target = 1 if j == jp else 0
                            residual += (val - target) ** 2
            
            if residual < 1e-10:
                return U, V, W
                
        return None

    # Try to find rank 11 for 2x3x2
    U232, V232, W232 = find_decomposition_232(11)
    
    if U232 is not None:
        # Construct full decomposition of rank 33
        rank = 33
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # B is 3x6. Split into 3 blocks of 3x2.
        # B = [B1 | B2 | B3]
        # C = [A*B1 | A*B2 | A*B3]
        # Block q (0, 1, 2) corresponds to columns 2q, 2q+1 of C.
        # And rows 0..2 of B (all rows, since B is 3x2 per block).
        
        for q in range(3):
            # Offset in V and W
            # V indices: j*p + k.
            # For block q, k = 2q, 2q+1.
            # V[j*p + k, col]
            # The local V232 has indices j*2 + k_local.
            # k_local = 0, 1 corresponds to k = 2q, 2q+1.
            
            # W indices: k*n + i.
            # For block q, k = 2q, 2q+1.
            # W[k*n + i, col]
            # Local W232 has indices k_local*n + i.
            
            for r in range(11):
                col = q * 11 + r
                
                # U is the same for all blocks?
                # A is 2x3. It is the same for all blocks.
                # So U[i*3+j, col] = U232[i*3+j, r]
                U[:, col] = U232[:, r]
                
                # V
                for j in range(3):
                    for k_local in range(2):
                        k = 2*q + k_local
                        V[j*6 + k, col] = V232[j*2 + k_local, r]
                
                # W
                for i in range(2):
                    for k_local in range(2):
                        k = 2*q + k_local
                        W[k*2 + i, col] = W232[k_local*2 + i, r]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook rank 36
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
