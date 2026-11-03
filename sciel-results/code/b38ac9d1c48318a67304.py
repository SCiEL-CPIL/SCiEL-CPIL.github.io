"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import random

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank34(n, m, p):
    """Build the known rank-34 solution using Strassen for 2x2x2 blocks."""
    strassen_terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,1],[1,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,1],[0,0]]),
        ([[0,0],[0,1]], [[-1,1],[0,0]], [[1,-1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[1,0],[0,0]]),
        ([[ -1,0],[1,0]], [[1,0],[1,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,1],[0,1]], [[1,0],[1,0]]),
    ]
    
    total_rank = 34
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        for (Ur, Vr, Wr) in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if Ur[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj)][col] += Ur[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if Vr[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk)][col] += Vr[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if Wr[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii)][col] += Wr[kk][ii]
            col += 1
    
    def add_schoolbook_block(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_schoolbook_block([0,1], [2], [0,1])
    add_schoolbook_block([0,1], [2], [2,3])
    add_schoolbook_block([2], [0,1], [0,1])
    add_schoolbook_block([2], [0,1], [2,3])
    add_schoolbook_block([2], [2], [0,1])
    add_schoolbook_block([2], [2], [2,3])
    
    return U, V, W

def _try_alternating_solve(r, n, m, p, max_iter=50, seed=0, tol=1e-10):
    """
    Try to find a rank-r decomposition using alternating least squares.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    U = np_rng.randn(n*m, r) * 0.5
    V = np_rng.randn(m*p, r) * 0.5
    W = np_rng.randn(p*n, r) * 0.5
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Build list of (u_idx, v_idx, w_idx) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m+j, j*p+k, k*n+i))
    
    for iteration in range(max_iter):
        # Solve for U given V, W
        # For each u_idx, we need sum_r U[u_idx,r] * (sum over triples involving u_idx of V[v_idx,r]*W[w_idx,r]) = 1
        # This is a linear system: for each u_idx, A_u * U[u_idx,:] = b_u
        # where A_u[r1, r2] = sum_{(u_idx, v_idx, w_idx) in triples} V[v_idx,r1]*W[w_idx,r2]
        # and b_u = number of triples involving u_idx (since target is 1 for each)
        
        # Group triples by u_idx
        from collections import defaultdict
        u_triples = defaultdict(list)
        for (ui, vi, wi) in triples:
            u_triples[ui].append((vi, wi))
        
        # Build system for U
        # For each u_idx: sum_r U[u_idx,r] * (sum_{(vi,wi) in u_triples[u_idx]} V[vi,r]*W[wi,r]) = len(u_triples[u_idx])
        # Let M_u[r] = sum_{(vi,wi) in u_triples[u_idx]} V[vi,r]*W[wi,r]
        # Then sum_r U[u_idx,r] * M_u[r] = len(u_triples[u_idx])
        # This is a single equation per u_idx, but we have r unknowns. Underdetermined.
        # We need more equations. The constraint is for each triple.
        # Actually, the constraint is: for each triple (ui, vi, wi), sum_r U[ui,r]*V[vi,r]*W[wi,r] = 1
        # This gives n*m*p equations in n*m*r unknowns for U (with V,W fixed).
        
        # Let's set up the linear system properly.
        # Variables: U[ui, r] for all ui, r
        # Equations: for each triple (ui, vi, wi): sum_r U[ui, r] * V[vi, r] * W[wi, r] = 1
        
        # This is a linear system A * x = b where x is vectorized U
        # A has n*m*p rows and n*m*r columns
        # For row corresponding to triple t=(ui, vi, wi), and column corresponding to (ui', r'):
        # A[t, (ui', r')] = V[vi, r'] * W[wi, r'] if ui' == ui, else 0
        
        # This is a block-diagonal-ish system. For each ui, the equations involving ui are:
        # For each (vi, wi) such that (ui, vi, wi) is a triple: sum_r U[ui,r]*V[vi,r]*W[wi,r] = 1
        # This gives m*p equations for each ui (since for fixed ui, vi ranges over m and wi over p)
        # So we have m*p equations and r unknowns for each ui.
        
        # For fixed ui, let's define:
        # For each (vi, wi): sum_r U[ui,r] * (V[vi,r]*W[wi,r]) = 1
        # Let c_{vi,wi}[r] = V[vi,r]*W[wi,r]
        # Then sum_r U[ui,r]*c_{vi,wi}[r] = 1 for each (vi,wi)
        # This is a system: C_ui * U[ui,:]^T = ones(m*p)
        # where C_ui[vi*p+wi, r] = V[vi,r]*W[wi,r]
        
        # If m*p > r, this is overdetermined. Use least squares.
        
        for ui in range(n*m):
            # Build C_ui
            C = np.zeros((m*p, r))
            for vi in range(m*p):
                for r_idx in range(r):
                    C[vi, r_idx] = V[vi, r_idx] * W[vi, r_idx]  # Wait, this is wrong.
            # Actually, for triple (ui, vi, wi), the coefficient is V[vi,r]*W[wi,r]
            # But vi and wi are independent indices. For fixed ui, the triples are (ui, vi, wi) for all vi, wi.
            # So for each (vi, wi) pair, we have one equation.
            # Let me reindex: for fixed ui, the equations are indexed by (vi, wi)
            # C[vi*p+wi, r] = V[vi, r] * W[wi, r]
            
            C = np.zeros((m*p, r))
            for vi in range(m):
                for wi in range(p):
                    for r_idx in range(r):
                        C[vi*p+wi, r_idx] = V[vi, r_idx] * W[wi, r_idx]
            
            b = np.ones(m*p)
            # Solve C * x = b in least squares
            try:
                sol, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
                U[ui, :] = sol
            except:
                pass
        
        # Similarly solve for V
        v_triples = defaultdict(list)
        for (ui, vi, wi) in triples:
            v_triples[vi].append((ui, wi))
        
        for vi in range(m*p):
            C = np.zeros((n*p, r))
            for ui in range(n):
                for wi in range(p):
                    for r_idx in range(r):
                        C[ui*p+wi, r_idx] = U[ui, r_idx] * W[wi, r_idx]
            b = np.ones(n*p)
            try:
                sol, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
                V[vi, :] = sol
            except:
                pass
        
        # Similarly solve for W
        w_triples = defaultdict(list)
        for (ui, vi, wi) in triples:
            w_triples[wi].append((ui, vi))
        
        for wi in range(p*n):
            C = np.zeros((n*m, r))
            for ui in range(n):
                for vi in range(m):
                    for r_idx in range(r):
                        C[ui*m+vi, r_idx] = U[ui, r_idx] * V[vi, r_idx]
            b = np.ones(n*m)
            try:
                sol, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
                W[wi, :] = sol
            except:
                pass
        
        # Check convergence
        max_err = 0.0
        for (ui, vi, wi) in triples:
            s = 0.0
            for r_idx in range(r):
                s += U[ui, r_idx] * V[vi, r_idx] * W[wi, r_idx]
            max_err = max(max_err, abs(s - 1.0))
        
        if max_err < tol:
            # Found a solution. Try to round to half-integers.
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            valid = True
            for (ui, vi, wi) in triples:
                s = 0.0
                for r_idx in range(r):
                    s += U_rounded[ui, r_idx] * V_rounded[vi, r_idx] * W_rounded[wi, r_idx]
                if abs(s - 1.0) > 1e-8:
                    valid = False
                    break
            
            if valid:
                return U_rounded, V_rounded, W_rounded
    
    return None

def _try_gaussian_elimination(r, n, m, p, num_samples=10, seed=0):
    """
    Try to find a rank-r decomposition by solving a linear system.
    Sample random integer matrices A (n x m) and B (m x p), compute C = A @ B,
    and set up linear equations for the coefficients.
    """
    rng = np.random.RandomState(seed)
    
    # Number of unknowns: r * (n*m + m*p + p*n)
    num_unknowns = r * (n*m + m*p + p*n)
    
    # Number of equations: num_samples * n * p (one per entry of C for each sample)
    num_eqs = num_samples * n * p
    
    if num_eqs < num_unknowns:
        # Not enough equations. Add more samples.
        num_samples = max(num_samples, (num_unknowns + n*p - 1) // (n*p))
        num_eqs = num_samples * n * p
    
    A_mat = np.zeros((num_eqs, num_unknowns))
    b_vec = np.zeros(num_eqs)
    
    sample_idx = 0
    for s in range(num_samples):
        A_sample = rng.randint(-3, 4, size=(n, m))
        B_sample = rng.randint(-3, 4, size=(m, p))
        C_sample = A_sample @ B_sample
        
        for i in range(n):
            for k in range(p):
                eq_idx = s * n * p + i * p + k
                
                # Equation: sum_{j=0}^{m-1} A[i,j]*B[j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # But U, V, W are the unknowns. The left side is C_sample[i,k].
                # The right side is a cubic in the unknowns, not linear!
                # This approach doesn't work directly because the tensor identity is cubic.
        
        sample_idx += 1
    
    # This approach is flawed because the identity is cubic, not linear.
    # We need a different strategy.
    return None

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # First, try to find a lower-rank solution
    best_result = None
    best_rank = 34
    
    # Try ranks from 30 to 33
    for r in range(30, 34):
        if time.time() - t0 > DEADLINE - 30:
            break
        
        for seed in range(20):
            if time.time() - t0 > DEADLINE - 30:
                break
            
            result = _try_alternating_solve(r, n, m, p, max_iter=30, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                if _validate(U_list, V_list, W_list, n, m, p):
                    best_result = (U_list, V_list, W_list)
                    best_rank = r
                    break
        
        if best_rank <= r:
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to rank-34 solution
    U, V, W = _build_rank34(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
