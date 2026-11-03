"""Matrix multiplication 3,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

def construct():
    import time
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n + i, r] follows the official trace convention
    
    # Start with schoolbook rank = 60
    # Try to find lower rank using structured decomposition
    
    # Known: for 3x4x5, we can potentially do better than 60
    # Strategy: group the schoolbook terms and look for cancellations
    
    # Let's try a direct approach: use the fact that some products can be shared
    # For 3x4x5, try rank 50-55
    
    # Use a known technique: if we have terms like (a1+a2)(b1+b2) = a1b1 + a1b2 + a2b1 + a2b2
    # We can sometimes combine 4 terms into fewer using clever combinations
    
    # For now, let's try to find a rank-55 decomposition by optimization
    
    # Actually, let me use a simpler but effective approach:
    # Start from schoolbook and try to merge terms where possible
    
    # The multiplication tensor for 3x4x5 has 60 non-zero entries in schoolbook.
    # We look for rank-r decomposition where r < 60.
    
    # Use numpy for numerical optimization, then round to half-integers
    
    import numpy as np
    from fractions import Fraction
    
    # Build the multiplication tensor T[i, j, k] where:
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i,j,k] = 1 if this is a valid product term, else 0
    # Actually, the tensor is: T[i, j, k] = delta (it's 1 for each (i,j,k) triple)
    # Wait, the matrix multiplication tensor is:
    # For C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k (it's the identity in a sense)
    # No, the tensor is such that:
    # T[i, j, k] contributes to C[i,k] via A[i,j]*B[j,k]
    # So the tensor is: T[i, j, k] = 1 (all ones in the valid range)
    
    # For rank-r decomposition:
    # T[i,j,k] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor is a 3-tensor T with indices (i, j, k):
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-1 decomposition term is: u_i * v_j * w_k
    # So we need: T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
    # 
    # But the interface uses U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # This means U, V, W are already flattened.
    
    # Let me re-read the interface:
    # U: (n*m, rank) - indexed by (i*m + j, r)
    # V: (m*p, rank) - indexed by (j*p + k, r)  
    # W: (p*n, rank) - indexed by (k*n + i, r)
    
    # The decomposition is:
    # For the multiplication tensor, we need:
    # delta_{i',i} * delta_{j',j} * delta_{k',k} ... 
    # 
    # Actually, the standard formulation is:
    # The multiplication tensor T is defined by:
    # T[i, j, k] = 1 for all valid (i, j, k)
    # 
    # And a rank-r decomposition gives:
    # T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # Wait, I think the convention is different. Let me think again.
    
    # For matrix multiplication C = A * B:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # The tensor T has T[i, j, k] such that:
    # <T, A, B> = C where the contraction is over j
    # 
    # A rank-1 term is: (a_i * b_j * c_k) where this contributes to C[i,k]
    # via a_i * c_k * (sum over j of b_j * ... )
    # 
    # I think the correct interpretation is:
    # T[i, j, k] = 1 (the tensor is all ones in the valid domain)
    # 
    # And we decompose: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # Where U[i,j,r] = U[i*m+j, r], etc.
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    # This is a system of n*m*p = 60 equations.
    
    # For schoolbook (rank 60):
    # For each (i,j,k), have one term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    
    # To get lower rank, we need to find r < 60 such that the system has a solution.
    
    # Let's try rank 55 first, using numerical methods then rounding.
    
    def try_rank(rank):
        """Try to find a rank-r decomposition."""
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # Total variables: (n*m + m*p + p*n) * rank = (12 + 20 + 15) * rank = 47 * rank
        # Constraints: n*m*p = 60 equations
        
        # This is underdetermined for rank >= 2, so solutions exist.
        # We need to find a solution with small integer/half-integer entries.
        
        # Use a structured approach: start from schoolbook and perturb
        
        # Actually, for exact solutions, let's use the following trick:
        # If we can find any real solution, we can try to find a rational one.
        
        # Let's use a random initialization and solve the linear system for each term.
        
        # The equations are:
        # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is nonlinear (product of three variables).
        
        # Alternative: fix two matrices and solve for the third.
        # If we fix U and V, then for each (i,j,k):
        # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
        # This is linear in W!
        
        # So we can do alternating least squares:
        # 1. Initialize U, V, W randomly
        # 2. Fix U, V, solve for W (linear system)
        # 3. Fix V, W, solve for U (linear system)
        # 4. Fix W, U, solve for V (linear system)
        # 5. Repeat until convergence
        
        # Then try to round to half-integers.
        
        np.random.seed(42)
        
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for iteration in range(200):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Solve for W: for each (i,j,k), sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
            # Group by (k, i): for fixed (k,i), we have equations for all j:
            # sum_j ... wait, no. For each (i,j,k), it's a separate equation.
            # But W only depends on (k,i), not on j.
            # So for fixed (k,i), the equations are:
            # For each j: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
            # This is: W[k*n+i, :] @ (U[i*m+j, :] * V[j*p+k, :]) = 1 for all j
            
            # So for each (k,i), we have m equations (one for each j) and rank unknowns.
            # If m <= rank, this is underdetermined.
            
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # Build the system: A @ W[k*n+i, :] = b
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # b[j] = 1
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    # Solve A @ w = b (least squares if m < rank)
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Solve for U: for each (i,j), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # For fixed (i,j), we have p equations (one for each k) and rank unknowns.
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Solve for V: for each (j,k), sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
            # For fixed (j,k), we have n equations (one for each i) and rank unknowns.
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
        
        # Check if the solution is valid
        def check_solution(U, V, W):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1) > 1e-6:
                            return False
            return True
        
        if not check_solution(U, V, W):
            return None
        
        # Try to round to half-integers
        def to_half_integer(x):
            return round(x * 2) / 2
        
        U_hi = np.array([[to_half_integer(x) for x in row] for row in U])
        V_hi = np.array([[to_half_integer(x) for x in row] for row in V])
        W_hi = np.array([[to_half_integer(x) for x in row] for row in W])
        
        # Check if the half-integer solution is valid
        def check_half_solution(Uh, Vh, Wh):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(Uh[i*m+j, :] * Vh[j*p+k, :] * Wh[k*n+i, :])
                        if abs(val - 1) > 1e-9:
                            return False
            return True
        
        if check_half_solution(U_hi, V_hi, W_hi):
            return (U_hi.tolist(), V_hi.tolist(), W_hi.tolist())
        
        # Try rounding to integers
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
        if check_half_solution(U_int, V_int, W_int):
            return (U_int.tolist(), V_int.tolist(), W_int.tolist())
        
        return None
    
    # Try ranks from 50 to 60
    for rank in range(50, 61):
        if time.time() - start_time > DEADLINE - 10:
            break
        result = try_rank(rank)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
