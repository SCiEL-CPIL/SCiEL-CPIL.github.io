import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    # Try to construct rank-25 decomposition directly
    # Use the block structure: split B into B1 (3x2) and B2 (3x3)
    # A1 (2x2) * B1 (2x2): rank 7 (Strassen)
    # A1 (2x2) * B2 (2x3): rank 6 (use 3 rank-2 terms)
    # A2 (2x1) * B2 (1x3): rank 3 (schoolbook)
    # Total: 7 + 6 + 3 = 16? That seems too low.
    
    # Actually, let me think again. The correct decomposition:
    # C = A * B where A is 2x3, B is 3x5
    # Split columns of A: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Split columns of B: B = [B1 | B2] where B1 is 3x2, B2 is 3x3
    # Then C = A1*B1 + A2*B2 (but dimensions don't match for addition)
    
    # Correct: C[:, 0:2] = A[:, 0:2] * B[0:2, :] + A[:, 2] * B[2, :]
    # Wait, that's not right either.
    
    # Let me use explicit indices:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split j into j=0,1 and j=2:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Group terms:
    # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    #         = (A1[i,:2] * B1[:2,k]) + A2[i,2]*B2[2,k]
    
    # Where A1 is 2x2, B1 is 2x5, A2 is 2x1, B2 is 1x5
    
    # For the first part: A1 (2x2) * B1 (2x5)
    # This can be seen as 5 separate 2x1 multiplications, or use Strassen-like
    # Actually, 2x2 times 2xp has rank at most 4p by schoolbook
    # But we can do better.
    
    # Known result: rank of 2x3x5 is 25.
    # Let me try to find it via optimization with rank 25.
    
    deadline = time.time() + 140  # Safety margin
    
    # First, let's try to verify if rank 25 is achievable by solving
    target_rank = 25
    
    def residual(params):
        size_u = nm * target_rank
        size_v = mp * target_rank
        size_w = pn * target_rank
        
        U = params[:size_u].reshape(nm, target_rank)
        V = params[size_u:size_u+size_v].reshape(mp, target_rank)
        W = params[size_u+size_v:].reshape(pn, target_rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                val = np.sum(U[i*m+j, :] * V[j_prime*p+k, :] * W[k_prime*n+i_prime, :])
                                if i == i_prime and j == j_prime and k == k_prime:
                                    res.append(val - 1)
                                else:
                                    res.append(val)
        return np.array(res)
    
    best_result = None
    best_rank = n * m * p  # 30
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        np.random.seed(42 + attempt)
        U0 = np.random.randn(nm, target_rank)
        V0 = np.random.randn(mp, target_rank)
        W0 = np.random.randn(pn, target_rank)
        
        U0 /= np.sqrt(target_rank)
        V0 /= np.sqrt(target_rank)
        W0 /= np.sqrt(target_rank)
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            sol = least_squares(residual, params0, method='lm', max_nfev=500, ftol=1e-10)
            if sol.success and sol.cost < 1e-8:
                U_sol = sol.x[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_sol = sol.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
        except Exception:
            pass
            
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

def _verify_decomposition(U, V, W, n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    R = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            s = np.sum(U[i*m+j, :] * V[j_prime*p+k, :] * W[k_prime*n+i_prime, :])
                            if i == i_prime and j == j_prime and k == k_prime:
                                if abs(s - 1) > 1e-9:
                                    return False
                            else:
                                if abs(s) > 1e-9:
                                    return False
    return True
