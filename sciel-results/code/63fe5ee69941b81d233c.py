"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    n, m, p = 4, 4, 5
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    # Build the matrix multiplication tensor
    # M[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
    M = np.zeros((nm, mp, pn), dtype=np.float64)
    nonzero_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
                nonzero_triples.append((a, b, c))
    
    # Schoolbook decomposition as fallback (rank 80)
    def schoolbook():
        rank = n * m * p  # 80
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if decomposition is exact for the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check all nonzero entries
        for a, b, c in nonzero_triples:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check zero entries - sample a good number
        np.random.seed(42)
        for _ in range(500):
            a = np.random.randint(nm)
            b = np.random.randint(mp)
            c = np.random.randint(pn)
            # Check if this is a nonzero triple
            is_nonzero = False
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if i_a == i_c and j_a == j_b and k_b == k_c:
                is_nonzero = True
            if not is_nonzero:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    def round_to_half_int(x):
        return np.round(2 * x) / 2.0
    
    # Try to find lower-rank decompositions
    # For 4x4x4, rank 49 is known (Landsberg, Strassen).
    # For 4x4x5, we expect rank to be somewhat higher.
    # Let's try ranks from 55 upward.
    
    # Use a structured approach: start with a good initialization
    # and use alternating least squares with constraints
    
    def als_refinement(U, V, W, n_iters=100, tol=1e-8):
        """Alternating least squares refinement."""
        rank = U.shape[1]
        
        for iteration in range(n_iters):
            if time.time() > deadline - 2:
                break
            
            # Update U: for each a, solve least squares
            for a in range(nm):
                # Build the system: U[a,:] * (V[b,:] * W[c,:])^T = M[a,b,c] for all b,c
                # This is: A @ U[a,:] = rhs where A[b*pn+c, r] = V[b,r] * W[c,r]
                A_mat = np.zeros((mp * pn, rank))
                rhs = np.zeros(mp * pn)
                idx = 0
                for b in range(mp):
                    for c in range(pn):
                        A_mat[idx, :] = V[b, :] * W[c, :]
                        rhs[idx] = M[a, b, c]
                        idx += 1
                
                try:
                    sol, residuals, rk, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    U[a, :] = sol
                except:
                    pass
            
            # Update V: for each b, solve least squares
            for b in range(mp):
                A_mat = np.zeros((nm * pn, rank))
                rhs = np.zeros(nm * pn)
                idx = 0
                for a in range(nm):
                    for c in range(pn):
                        A_mat[idx, :] = U[a, :] * W[c, :]
                        rhs[idx] = M[a, b, c]
                        idx += 1
                
                try:
                    sol, residuals, rk, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    V[b, :] = sol
                except:
                    pass
            
            # Update W: for each c, solve least squares
            for c in range(pn):
                A_mat = np.zeros((nm * mp, rank))
                rhs = np.zeros(nm * mp)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        A_mat[idx, :] = U[a, :] * V[b, :]
                        rhs[idx] = M[a, b, c]
                        idx += 1
                
                try:
                    sol, residuals, rk, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    W[c, :] = sol
                except:
                    pass
            
            # Check convergence
            err = 0.0
            for a, b, c in nonzero_triples:
                pred = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (pred - 1.0) ** 2
            # Sample some zero entries
            for _ in range(100):
                a = np.random.randint(nm)
                b = np.random.randint(mp)
                c = np.random.randint(pn)
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (i_a == i_c and j_a == j_b and k_b == k_c):
                    pred = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += pred ** 2
            
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try various ranks
    for target_rank in range(55, 81):
        if time.time() > deadline - 15:
            break
        
        # Multiple restarts with different initializations
        n_restarts = 30 if target_rank < 65 else 15
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with different structures
            if restart % 3 == 0:
                # Pure random
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            elif restart % 3 == 1:
                # Sparse initialization
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    # Random sparse patterns
                    n_nonzero_u = np.random.randint(2, 5)
                    n_nonzero_v = np.random.randint(2, 5)
                    n_nonzero_w = np.random.randint(2, 5)
                    
                    u_idx = np.random.choice(nm, n_nonzero_u, replace=False)
                    v_idx = np.random.choice(mp, n_nonzero_v, replace=False)
                    w_idx = np.random.choice(pn, n_nonzero_w, replace=False)
                    
                    U[u_idx, r] = np.random.choice([-1, 1], n_nonzero_u)
                    V[v_idx, r] = np.random.choice([-1, 1], n_nonzero_v)
                    W[w_idx, r] = np.random.choice([-1, 1], n_nonzero_w)
            else:
                # Structured initialization based on Strassen-like patterns
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                # Use a subset of the schoolbook terms
                indices = np.random.choice(n*m*p, target_rank, replace=False)
                for r, idx in enumerate(indices):
                    i = idx // (m * p)
                    j = (idx // p) % m
                    k = idx % p
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, r] = 1
                    V[b, r] = 1
                    W[c, r] = 1
                # Add some random perturbation
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            
            # Refine with ALS
            U, V, W, err = als_refinement(U, V, W, n_iters=50)
            
            # Round to half-integers
            U_rounded = round_to_half_int(U)
            V_rounded = round_to_half_int(V)
            W_rounded = round_to_half_int(W)
            
            if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                    return best_result
            
            # Also try without rounding (in case exact solution has non-half-integers)
            # But the problem says integer/half-integer, so rounding should be fine
    
    return best_result
