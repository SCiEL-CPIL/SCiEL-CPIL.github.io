"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Indexing convention:
    # A is n x m, index a = i*m + j where i in [0,n), j in [0,m)
    # B is m x p, index b = j*p + k where j in [0,m), k in [0,p)
    # C is n x p, index c = k*n + i where k in [0,p), i in [0,n)  (trace convention W[k*n+i, r])
    
    # Tensor T[a, b, c] = 1 if there exists j such that:
    #   a = i*m + j, b = j'*p + k, c = k'*n + i'
    #   with j = j' (inner dimension match) and i = i' and k = k'
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[a,b,c] = 1 when a = i*m+j, b = j*p+k, c = k*n+i (same j, i, k)
    
    # Build the support set
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    support_set = set(support)
    
    # Schoolbook decomposition (rank = n*m*p = 20)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for idx, (a, b, c) in enumerate(support):
        U_school[a, idx] = 1.0
        V_school[b, idx] = 1.0
        W_school[c, idx] = 1.0
    
    def to_list(arr):
        return arr.tolist()
    
    def compute_error(U, V, W):
        max_err = 0.0
        all_tuples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
        for (a, b, c) in all_tuples:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if (a, b, c) in support_set:
                err = abs(val - 1.0)
            else:
                err = abs(val)
            if err > max_err:
                max_err = err
        return max_err
    
    # Try to find lower rank decompositions using alternating least squares
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    deadline = time.time() + 110  # 110s safety margin
    
    # Lower bound for rank: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    # Also, for matrix multiplication, known lower bounds apply
    # Try ranks from 10 upward
    for r in range(10, rank_school):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 30 if r <= 14 else 15
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with some structure
            np.random.seed(42 + r * 100 + attempt)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Alternating minimization with full constraint set
            for iteration in range(300):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each a, solve least squares
                for a in range(n*m):
                    rows = []
                    targets = []
                    for (a2, b, c) in support:
                        if a2 == a:
                            rows.append(V[b, :] * W[c, :])
                            targets.append(1.0)
                    # Add non-support constraints
                    for b in range(m*p):
                        for c in range(p*n):
                            if (a, b, c) not in support_set:
                                rows.append(V[b, :] * W[c, :])
                                targets.append(0.0)
                    
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[a, :] = sol
                
                # Update V: for each b, solve least squares
                for b in range(m*p):
                    rows = []
                    targets = []
                    for (a, b2, c) in support:
                        if b2 == b:
                            rows.append(U[a, :] * W[c, :])
                            targets.append(1.0)
                    for a in range(n*m):
                        for c in range(p*n):
                            if (a, b, c) not in support_set:
                                rows.append(U[a, :] * W[c, :])
                                targets.append(0.0)
                    
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[b, :] = sol
                
                # Update W: for each c, solve least squares
                for c in range(p*n):
                    rows = []
                    targets = []
                    for (a, b, c2) in support:
                        if c2 == c:
                            rows.append(U[a, :] * V[b, :])
                            targets.append(1.0)
                    for a in range(n*m):
                        for b in range(m*p):
                            if (a, b, c) not in support_set:
                                rows.append(U[a, :] * V[b, :])
                                targets.append(0.0)
                    
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[c, :] = sol
                
                # Check convergence
                err = compute_error(U, V, W)
                if err < 1e-10:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < rank_school:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
# EVOLVE-BLOCK-END
