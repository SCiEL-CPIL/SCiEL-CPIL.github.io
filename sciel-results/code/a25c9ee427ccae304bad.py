import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT

    # Build target tensor for validation
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_tensor[u, v, w] = 1.0

    def to_half_int_list(matrix):
        scaled = np.rint(matrix * 2.0)
        result = scaled / 2.0
        return [[float(v) for v in row] for row in result]

    def validate(U, V, W):
        """Check if U, V, W decompose the tensor exactly."""
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, target_tensor, atol=1e-8)

    def build_rank_17():
        """Construct rank-17 decomposition using Strassen for 2x2 core + schoolbook."""
        rank = 17
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U[0, col] = 1.0; U[4, col] = 1.0
        V[0, col] = 1.0; V[4, col] = 1.0
        W[0, col] = 1.0; W[3, col] = 1.0
        col += 1
        
        # M2 = (A[1,0]+A[1,1])*(B[0,0]+B[0,1])
        U[3, col] = 1.0; U[4, col] = 1.0
        V[0, col] = 1.0; V[1, col] = 1.0
        W[2, col] = 1.0; W[1, col] = 1.0
        col += 1
        
        # M3 = (A[0,0]-A[1,1])*(B[1,0]+B[1,1])
        U[0, col] = 1.0; U[4, col] = -1.0
        V[3, col] = 1.0; V[4, col] = 1.0
        W[3, col] = 1.0
        col += 1
        
        # M4 = A[0,0]*(B[0,1]-B[1,1])
        U[0, col] = 1.0
        V[1, col] = 1.0; V[4, col] = -1.0
        W[0, col] = 1.0; W[2, col] = 1.0; W[3, col] = -1.0
        col += 1
        
        # M5 = A[1,1]*(B[1,0]-B[0,0])
        U[4, col] = 1.0
        V[3, col] = 1.0; V[0, col] = -1.0
        W[0, col] = -1.0; W[1, col] = 1.0
        col += 1
        
        # M6 = (A[0,0]+A[0,1])*(B[1,1]-B[0,1])
        U[0, col] = 1.0; U[1, col] = 1.0
        V[4, col] = 1.0; V[1, col] = -1.0
        W[2, col] = -1.0; W[1, col] = -1.0; W[3, col] = 1.0
        col += 1
        
        # M7 = (A[1,0]+A[1,1])*(B[0,0]-B[1,0])
        U[3, col] = 1.0; U[4, col] = 1.0
        V[0, col] = 1.0; V[3, col] = -1.0
        W[0, col] = 1.0; W[2, col] = -1.0; W[1, col] = -1.0
        col += 1
        
        # Term 8: A[0,2]*B[2,0] -> C[0,0]
        U[2, col] = 1.0; V[6, col] = 1.0; W[0, col] = 1.0
        col += 1
        
        # Term 9: A[0,2]*B[2,1] -> C[0,1]
        U[2, col] = 1.0; V[7, col] = 1.0; W[2, col] = 1.0
        col += 1
        
        # Term 10: A[0,0]*B[0,2] -> C[0,2]
        U[0, col] = 1.0; V[2, col] = 1.0; W[4, col] = 1.0
        col += 1
        
        # Term 11: A[0,1]*B[1,2] -> C[0,2]
        U[1, col] = 1.0; V[5, col] = 1.0; W[4, col] = 1.0
        col += 1
        
        # Term 12: A[0,2]*B[2,2] -> C[0,2]
        U[2, col] = 1.0; V[8, col] = 1.0; W[4, col] = 1.0
        col += 1
        
        # Term 13: A[1,2]*B[2,0] -> C[1,0]
        U[5, col] = 1.0; V[6, col] = 1.0; W[1, col] = 1.0
        col += 1
        
        # Term 14: A[1,2]*B[2,1] -> C[1,1]
        U[5, col] = 1.0; V[7, col] = 1.0; W[3, col] = 1.0
        col += 1
        
        # Term 15: A[1,0]*B[0,2] -> C[1,2]
        U[3, col] = 1.0; V[2, col] = 1.0; W[5, col] = 1.0
        col += 1
        
        # Term 16: A[1,1]*B[1,2] -> C[1,2]
        U[4, col] = 1.0; V[5, col] = 1.0; W[5, col] = 1.0
        col += 1
        
        # Term 17: A[1,2]*B[2,2] -> C[1,2]
        U[5, col] = 1.0; V[8, col] = 1.0; W[5, col] = 1.0
        col += 1
        
        return U, V, W

    def build_schoolbook():
        """Construct schoolbook rank-18 decomposition."""
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        return U_sb, V_sb, W_sb

    def try_rank_16_search():
        """Attempt to find a rank-16 decomposition via structured search."""
        if time.time() > deadline - 30:
            return None
        
        # Try a different Strassen-like decomposition that might save one term
        # Strategy: Try to merge some schoolbook terms with Strassen terms
        # by using non-integer coefficients
        
        # First, try a linear algebra approach:
        # Set up the system and try to solve for rank 16
        # This is a non-convex problem, so we'll try random initializations
        
        best_result = None
        
        # Try several random structured initializations
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 25:
                break
            
            rank = 16
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Try gradient-free optimization using coordinate descent
            # on the residual
            for step in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Compute residual
                result = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                residual = target_tensor - result
                
                if np.allclose(residual, 0, atol=1e-10):
                    return U, V, W
                
                # Update one factor at a time (coordinate descent)
                # Update U
                for c in range(rank):
                    # Optimal U[:,c] given V[:,c] and W[:,c]
                    # minimize ||T - sum_c U_c ⊗ V_c ⊗ W_c||
                    # For fixed c, the contribution of term c is U_c ⊗ V_c ⊗ W_c
                    # Residual without term c: R_c = T - sum_{c'!=c} U_{c'} ⊗ V_{c'} ⊗ W_{c'}
                    # We want to minimize ||R_c - U_c ⊗ V_c ⊗ W_c||
                    # This gives U_c = (R_c * (V_c ⊗ W_c)) / ||V_c ⊗ W_c||^2
                    # where * is the inner product over the other two indices
                    
                    # Compute R_c
                    R_c = residual.copy()
                    R_c += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                    
                    # Optimal U_c
                    denom = np.sum(V[:, c]**2) * np.sum(W[:, c]**2)
                    if denom < 1e-10:
                        continue
                    U[:, c] = np.sum(R_c * np.outer(V[:, c], W[:, c]), axis=(1,2)) / denom
                    
                # Update V
                for c in range(rank):
                    R_c = residual.copy()
                    R_c += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                    denom = np.sum(U[:, c]**2) * np.sum(W[:, c]**2)
                    if denom < 1e-10:
                        continue
                    V[:, c] = np.sum(R_c * np.outer(U[:, c], W[:, c]), axis=(0,2)) / denom
                    
                # Update W
                for c in range(rank):
                    R_c = residual.copy()
                    R_c += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                    denom = np.sum(U[:, c]**2) * np.sum(V[:, c]**2)
                    if denom < 1e-10:
                        continue
                    W[:, c] = np.sum(R_c * np.outer(U[:, c], V[:, c]), axis=(0,1)) / denom
                
                # Recompute residual
                result = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                residual = target_tensor - result
                
                if np.allclose(residual, 0, atol=1e-10):
                    return U, V, W
        
        return None

    # First try the known rank-17 solution
    U17, V17, W17 = build_rank_17()
    if validate(U17, V17, W17):
        # Try to find rank 16
        result_16 = try_rank_16_search()
        if result_16 is not None:
            U16, V16, W16 = result_16
            if validate(U16, V16, W16):
                return {"U": to_half_int_list(U16), "V": to_half_int_list(V16), "W": to_half_int_list(W16)}
        return {"U": to_half_int_list(U17), "V": to_half_int_list(V17), "W": to_half_int_list(W17)}
    
    # Fallback to schoolbook rank 18
    U_sb, V_sb, W_sb = build_schoolbook()
    return {"U": to_half_int_list(U_sb), "V": to_half_int_list(V_sb), "W": to_half_int_list(W_sb)}
