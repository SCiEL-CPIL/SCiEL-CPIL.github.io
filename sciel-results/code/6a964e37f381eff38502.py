"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]
TIME_LIMIT = 150
SAFETY_MARGIN = 5

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def check_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Build the multiplication tensor
        # T[a_idx, b_idx, c_idx] = 1 if the indices are consistent
        # a_idx = i*m + j, b_idx = j'*p + k, c_idx = k'*n + i'
        # T = 1 iff j == j' and k == k'
        
        # Compute the reconstructed tensor from the decomposition
        # Reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Check against the true tensor
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                a = i * m + j
                                b = jp * p + k
                                c = kp * n + ip
                                true_val = 1.0 if (j == jp and k == kp) else 0.0
                                recon_val = 0.0
                                for r in range(rank):
                                    recon_val += U[a, r] * V[b, r] * W[c, r]
                                if abs(true_val - recon_val) > 1e-9:
                                    return False
        return True
    
    # Try rank 28 first
    rank_candidate = 2 * p  # 28
    
    # Construction for rank 2*p:
    # For each k in [0,p) and each t in {0,1}:
    #   term index r = 2*k + t
    #   U[i*m+j, r] = delta(j, t) for all i  -- this picks A[i,t]
    #   V[jp*p+kp, r] = delta(jp, t) * delta(kp, k) -- this picks B[t,k]
    #   W[kp*n+ip, r] = delta(kp, k) for all ip -- this contributes to C[k,ip]
    #
    # Wait, this gives: sum_r U[a,r]*V[b,r]*W[c,r]
    # = sum_{k,t} delta(j,t) * delta(jp,t)*delta(kp,k) * delta(kp,k)
    # = sum_{k,t} delta(j,t)*delta(jp,t)*delta(kp,k)*delta(kp,k)
    # = sum_t delta(j,t)*delta(jp,t) * delta(kp,k)
    # = delta(j,jp) * delta(kp,k)
    #
    # That's exactly what we want! Let me verify:
    # a = i*m+j, b = jp*p+k, c = kp*n+ip
    # sum_r U[a,r]*V[b,r]*W[c,r]
    # = sum_{k'=0}^{p-1} sum_{t=0}^{1} U[i*m+j, 2k'+t] * V[jp*p+k, 2k'+t] * W[kp*n+ip, 2k'+t]
    # U[i*m+j, 2k'+t] = delta(j,t)
    # V[jp*p+k, 2k'+t] = delta(jp,t) * delta(k,k')
    # W[kp*n+ip, 2k'+t] = delta(kp,k')
    # So sum = sum_{k',t} delta(j,t)*delta(jp,t)*delta(k,k')*delta(kp,k')
    # = sum_t delta(j,t)*delta(jp,t) * delta(k,k')*delta(kp,k') summed over k'
    # = delta(j,jp) * delta(k,kp)  [since sum_{k'} delta(k,k')*delta(kp,k') = delta(k,kp)]
    # 
    # Yes! This works!
    
    rank = rank_candidate
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for t in range(m):  # t in {0, 1}
            r = 2 * k + t
            # U: for each i, U[i*m + j, r] = delta(j, t)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1 if j == t else 0
            # V: V[jp*p + kp, r] = delta(jp, t) * delta(kp, k)
            for jp in range(m):
                for kp in range(p):
                    V[jp * p + kp][r] = 1 if (jp == t and kp == k) else 0
            # W: W[kp*n + ip, r] = delta(kp, k)
            for kp in range(p):
                for ip in range(n):
                    W[kp * n + ip][r] = 1 if kp == k else 0
    
    # Verify the decomposition
    if check_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if something went wrong
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
