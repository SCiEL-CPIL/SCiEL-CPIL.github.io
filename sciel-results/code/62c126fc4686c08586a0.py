"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Build the target tensor in flattened form
    # T[(i,j), (j,k), (k,i)] = 1
    # Mode sizes: n*m, m*p, p*n
    size1 = n * m
    size2 = m * p
    size3 = p * n
    
    # Build target tensor as a list of (a,b,c) indices where T[a,b,c] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    # Try to find a low-rank decomposition
    # Start with a reasonable rank estimate
    # Lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Try ranks from low to high
    
    deadline = time.time() + 100  # 100 second budget, leave margin
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Initialize with schoolbook
    def make_schoolbook():
        U = np.zeros((size1, rank_schoolbook))
        V = np.zeros((size2, rank_schoolbook))
        W = np.zeros((size3, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to reduce rank using alternating minimization
    # For a given rank r, we want to find U (size1 x r), V (size2 x r), W (size3 x r)
    # such that for all (a,b,c) in target_indices: sum_d U[a,d]*V[b,d]*W[c,d] = 1
    # and for all other (a,b,c): sum_d U[a,d]*V[b,d]*W[c,d] = 0
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        # Check all target entries
        for (a, b, c) in target_indices:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check some random non-target entries
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, size1)
            b = np.random.randint(0, size2)
            c = np.random.randint(0, size3)
            # Check if (a,b,c) is a target
            is_target = (a, b, c) in set(target_indices)
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if is_target:
                if abs(val - 1.0) > tol:
                    return False
            else:
                if abs(val) > tol:
                    return False
        return True
    
    def alternating_minimize(U, V, W, n_iters=50):
        """Run alternating minimization to refine the decomposition."""
        r = U.shape[1]
        for it in range(n_iters):
            # Update U: for each row a, U[a,:] should satisfy
            # sum_d U[a,d] * (V[b,:]*W[c,:])_d = 1 for all (a,b,c) in target with this a
            # This is a linear system for each a
            for a in range(size1):
                # Collect all (b,c) pairs where (a,b,c) is a target
                eqs = []
                rhs = []
                for (ta, tb, tc) in target_indices:
                    if ta == a:
                        # U[a,:] dot (V[tb,:] * W[tc,:]) = 1
                        vec = V[tb, :] * W[tc, :]
                        eqs.append(vec)
                        rhs.append(1.0)
                if eqs:
                    A_mat = np.array(eqs)
                    b_vec = np.array(rhs)
                    # Solve least squares
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[a, :] = sol
            # Update V
            for b in range(size2):
                eqs = []
                rhs = []
                for (ta, tb, tc) in target_indices:
                    if tb == b:
                        vec = U[ta, :] * W[tc, :]
                        eqs.append(vec)
                        rhs.append(1.0)
                if eqs:
                    A_mat = np.array(eqs)
                    b_vec = np.array(rhs)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[b, :] = sol
            # Update W
            for c in range(size3):
                eqs = []
                rhs = []
                for (ta, tb, tc) in target_indices:
                    if tc == c:
                        vec = U[ta, :] * V[tb, :]
                        eqs.append(vec)
                        rhs.append(1.0)
                if eqs:
                    A_mat = np.array(eqs)
                    b_vec = np.array(rhs)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[c, :] = sol
        return U, V, W
    
    # Try various ranks
    for r in range(25, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(size1, r) * 0.1
            V = np.random.randn(size2, r) * 0.1
            W = np.random.randn(size3, r) * 0.1
            
            # Run alternating minimization
            U, V, W = alternating_minimize(U, V, W, n_iters=30)
            
            # Check if valid
            if check_decomposition(U, V, W, tol=1e-5):
                # Try to clean up to exact integer/half-integer values
                # Round to nearest 0.5
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                if check_decomposition(U_clean, V_clean, W_clean, tol=1e-9):
                    return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
                else:
                    # Return the non-cleaned version
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
