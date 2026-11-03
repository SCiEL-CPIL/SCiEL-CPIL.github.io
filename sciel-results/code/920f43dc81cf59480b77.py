import numpy as np
import time
from itertools import product
from scipy.optimize import minimize

# Precomputed Strassen patterns for 2x2x2 (verified exact)
STRASSEN_PATTERNS = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
    ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]

SCHOOLBOOK_222 = [
    ([1,0,0,0],[1,0,0,0],[1,0,0,0]),
    ([0,1,0,0],[0,1,0,0],[0,1,0,0]),
    ([0,0,1,0],[0,0,1,0],[0,0,1,0]),
    ([0,0,0,1],[0,0,0,1],[0,0,0,1]),
]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _tensor_contraction(U, V, W):
    """Compute the tensor from factor matrices.
    T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    """
    rank = U.shape[1]
    # Use einsum for efficient computation
    return np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)

def _residual(U, V, W, target_T):
    """Compute residual ||T - U*V*W||_F^2."""
    T_recon = _tensor_contraction(U, V, W)
    return np.sum((T_recon - target_T)**2)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Check diagonal entries = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U[a], V[b]*W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries = 0
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    continue
                val = np.dot(U[a], V[b]*W[c])
                if abs(val) > tol:
                    return False
    return True

def _build_strassen_rank36():
    """Build the known rank-36 Strassen-based decomposition."""
    n, m, p = 2, 4, 5
    nm, mp, pn = n*m, m*p, p*n
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)
    
    def add_222_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        for u_pat, v_pat, w_pat in STRASSEN_PATTERNS:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    U[(a_i_off + idx//2)*m + a_j_off + idx%2, col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    V[(b_j_off + idx//2)*p + b_k_off + idx%2, col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    W[(c_k_off + idx%2)*n + c_i_off + idx//2, col_off] = val
            col_off += 1
        return col_off
    
    def add_221_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Schoolbook for 2x2 * 2x1
        patterns_221 = [
            ([1,0,0,0],[1,0],[1,0]),
            ([0,1,0,0],[0,1],[1,0]),
            ([0,0,1,0],[1,0],[0,1]),
            ([0,0,0,1],[0,1],[0,1]),
        ]
        for u_pat, v_pat, w_pat in patterns_221:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    U[(a_i_off + idx//2)*m + a_j_off + idx%2, col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    V[(b_j_off + idx)*p + b_k_off, col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    W[(c_k_off)*n + c_i_off + idx, col_off] = val
            col_off += 1
        return col_off
    
    col = 0
    # A1(2x2) * B1(2x2) -> C11(2x2)
    col = add_222_block(0, 0, 0, 0, 0, 0, col)
    # A1(2x2) * B2(2x2) -> C12(2x2)
    col = add_222_block(0, 0, 0, 2, 0, 2, col)
    # A1(2x2) * B3(2x1) -> C13(2x1)
    col = add_221_block(0, 0, 0, 4, 0, 4, col)
    # A2(2x2) * B1(2x2) -> C21(2x2)
    col = add_222_block(0, 2, 2, 0, 0, 0, col)
    # A2(2x2) * B2(2x2) -> C22(2x2)
    col = add_222_block(0, 2, 2, 2, 0, 2, col)
    # A2(2x2) * B3(2x1) -> C23(2x1)
    col = add_221_block(0, 2, 2, 4, 0, 4, col)
    
    assert col == 36, f"Expected rank 36, got {col}"
    return U, V, W

def _random_init(nm, mp, pn, rank, seed):
    """Random initialization with proper scaling."""
    rng = np.random.RandomState(seed)
    scale = 0.5
    U = rng.randn(nm, rank) * scale
    V = rng.randn(mp, rank) * scale
    W = rng.randn(pn, rank) * scale
    return U, V, W

def _optimize_rank35(target_T, nm, mp, pn, deadline):
    """Try to find a rank-35 decomposition using optimization."""
    rank = 35
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        seed = 42 + restart
        U0, V0, W0 = _random_init(nm, mp, pn, rank, seed)
        
        # Flatten parameters
        x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        
        def objective(x):
            U_x = x[:nm*rank].reshape(nm, rank)
            V_x = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_x = x[nm*rank+mp*rank:].reshape(pn, rank)
            return _residual(U_x, V_x, W_x, target_T)
        
        def jac(x):
            U_x = x[:nm*rank].reshape(nm, rank)
            V_x = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_x = x[nm*rank+mp*rank:].reshape(pn, rank)
            
            # J_U[i,a] = sum_{j,k} T[i,j,k]*V[j,a]*W[k,a] - 2*sum_{j,k} (U[i,a]*V[j,a]*W[k,a])*V[j,a]*W[k,a]
            # Simpler: use automatic differentiation via finite differences is too slow
            # Use the fact that dT[i,j,k]/dU[i,a] = V[j,a]*W[k,a]
            # dL/dU[i,a] = sum_{j,k} 2*(T[i,j,k] - T_recon[i,j,k]) * V[j,a]*W[k,a]
            
            T_recon = _tensor_contraction(U_x, V_x, W_x)
            residual = target_T - T_recon
            
            dU = 2.0 * np.einsum('ijk,jk,a->ia', residual, V_x*W_x, np.ones(1))  # placeholder
            # Correct computation:
            dU = 2.0 * np.einsum('ijk,jr,kr->ir', residual, V_x, W_x, optimize=True)
            dV = 2.0 * np.einsum('ijk,ir,kr->jr', residual, U_x, W_x, optimize=True)
            dW = 2.0 * np.einsum('ijk,ir,jr->kr', residual, U_x, V_x, optimize=True)
            
            return np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=jac, 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30})
            
            if res.fun < best_residual:
                best_residual = res.fun
                U_best = res.x[:nm*rank].reshape(nm, rank)
                V_best = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_best = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                best_U, best_V, best_W = U_best, V_best, W_best
        
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # Try to polish the best solution
    if best_U is not None:
        if _verify_decomposition(best_U, best_V, best_W, 2, 4, 5, tol=1e-6):
            # Round to half-integers
            U_round = np.round(best_U * 2) / 2
            V_round = np.round(best_V * 2) / 2
            W_round = np.round(best_W * 2) / 2
            if _verify_decomposition(U_round, V_round, W_round, 2, 4, 5, tol=1e-9):
                return U_round, V_round, W_round, 35
    
    return None

def _try_groebner_rank35(deadline):
    """
    Attempt a Groebner basis approach to find rank-35 decomposition.
    This is a symbolic approach - we set up the polynomial system and try to solve it.
    """
    n, m, p = 2, 4, 5
    nm, mp, pn = n*m, m*p, p*n
    rank = 35
    
    # The polynomial system is:
    # For each (i,j,k) with i*m+j, j*p+k, k*n+i matching:
    #   sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # For all other (a,b,c) triples:
    #   sum_r U[a,r] * V[b,r] * W[c,r] = 0
    #
    # This is a system of trilinear equations. The number of unknowns is:
    # nm*rank + mp*rank + pn*rank = 8*35 + 20*35 + 10*35 = 280 + 700 + 350 = 1330
    # The number of equations is nm*mp*pn = 8*20*10 = 1600
    #
    # This is highly overdetermined. A Groebner basis approach is computationally infeasible
    # for this size. Instead, we'll use a smarter approach:
    #
    # Key insight: The matrix multiplication tensor for 2x4x5 has a known structure.
    # The rank is at most 36 (Strassen-based). Can we do better?
    #
    # One approach: Try to merge some of the rank-1 terms from the Strassen decomposition.
    # Another: Use a different block structure.
    #
    # Let's try a 2x4 * 4x5 decomposition where we treat it as:
    # - A is 2x4, B is 4x5, C is 2x5
    # - Split B into B1 (4x2) and B2 (4x3)
    # - C into C1 (2x2) and C2 (2x3)
    #
    # A * B1 = C1: 2x4 * 4x2 -> 2x2 (rank <= 7 with Strassen)
    # A * B2 = C2: 2x4 * 4x3 -> 2x3 (need to figure out rank)
    #
    # For 2x4 * 4x3:
    # Split into A1(2x2)*B21(2x3) + A2(2x2)*B22(2x3)
    # A1*B21: 2x2 * 2x3 -> 2x3
    # A2*B22: 2x2 * 2x3 -> 2x3
    #
    # For 2x2 * 2x3:
    # This is a 2x2x3 tensor. We can use Strassen-like methods.
    # Rank of 2x2x3 tensor: at most 4*3=12 schoolbook, but can we do better?
    #
    # Actually, let's think about this differently. The 2x2x3 multiplication:
    # A = [[a,b],[c,d]] (2x2), B = [[e,f,g],[h,i,j]] (2x3), C = [[a*e+b*h, a*f+b*i, a*g+b*j], [c*e+d*h, c*f+d*i, c*g+d*j]]
    #
    # This has 12 entries. The rank of this tensor... Let me think.
    #
    # Actually, the rank of the 2x2x2 tensor is 7 (Strassen). For 2x2x3, we can view it as:
    # 2x2x3 = 2x2x2 + 2x2x1
    # Rank <= 7 + 4 = 11
    #
    # So A*B2: 2x4*4x3 -> 2x3
    # = A1*B21 (2x2*2x3) + A2*B22 (2x2*2x3)
    # Rank <= 11 + 11 = 22
    #
    # Total: 7 + 22 = 29? That seems too optimistic. Let me reconsider.
    #
    # Wait, I need to be more careful. The issue is that when we split, the same A matrix
    # is used in multiple blocks, so we can't just add ranks independently.
    #
    # Let me reconsider the structure. The key constraint is that U must be the same across
    # all blocks that share the same (i,j) indices.
    
    # Alternative approach: Try a Winograd Fourier method
    # For 2x4x5, we can use polynomial evaluation.
    # 
    # A(x) = A0 + A1*x (where A0 is first 2 columns, A1 is next 2 columns of A)
    # B(x) = B0 + B1*x + B2*x^2 + B3*x^3 + B4*x^4 (5 columns of B)
    # C(x) = C0 + C1*x + ... + C3*x^3 (5 columns of C, but C is 2x5 so C has 5 columns)
    #
    # Wait, A is 2x4, B is 4x5, C is 2x5.
    # C = A * B
    #
    # If we view A as a 2x4 matrix, we can split it as A = [A_left(2x2) | A_right(2x2)]
    # B is 4x5, split as B = [B_top(2x5); B_bottom(2x5)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # This is the same as before. The rank is at most rank(A_left*B_top) + rank(A_right*B_bottom)
    # but with shared U factors.
    
    # Let's try a completely different approach: use the fact that the tensor is
    # the Kronecker product of smaller tensors.
    #
    # 2x4x5 = 2x(2x2)x5 = (2x2x5) x (1x2x1) approximately
    #
    # Actually, let's try to find a rank-35 solution by modifying the rank-36 solution.
    # We'll try to eliminate one rank term by expressing it as a combination of others.
    
    U36, V36, W36 = _build_strassen_rank36()
    
    # Try to find a linear dependency among the rank-1 terms
    # If there exists a vector (alpha, beta, gamma) such that:
    # alpha * U[:,r] + sum of adjustments = 0 for some r, and similarly for V and W
    #
    # More precisely, try to remove column r from U, V, W and see if we can
    # compensate by adjusting other columns.
    
    # For each column r, try to see if we can express it in terms of others
    for r_remove in range(35):
        if time.time() > deadline - 5:
            break
        
        # Remove column r_remove
        U_rem = np.delete(U36, r_remove, axis=1)
        V_rem = np.delete(V36, r_remove, axis=1)
        W_rem = np.delete(W36, r_remove, axis=1)
        
        # Check if the removed column is in the span of the remaining columns
        # for each matrix separately. If U[:,r_remove] is in span(U[:,others]),
        # and similarly for V and W, then we might be able to absorb it.
        
        # This is a rough heuristic - check if the removed column can be approximated
        # by a linear combination of the remaining columns.
        
        # Use least squares to find coefficients
        try:
            # For U
            if U_rem.shape[1] > 0:
                coeff_U, _, _, _ = np.linalg.lstsq(U_rem, U36[:, r_remove], rcond=None)
                U_err = np.linalg.norm(U36[:, r_remove] - U_rem @ coeff_U)
            else:
                U_err = np.linalg.norm(U36[:, r_remove])
            
            if V_rem.shape[1] > 0:
                coeff_V, _, _, _ = np.linalg.lstsq(V_rem, V36[:, r_remove], rcond=None)
                V_err = np.linalg.norm(V36[:, r_remove] - V_rem @ coeff_V)
            else:
                V_err = np.linalg.norm(V36[:, r_remove])
            
            if W_rem.shape[1] > 0:
                coeff_W, _, _, _ = np.linalg.lstsq(W_rem, W36[:, r_remove], rcond=None)
                W_err = np.linalg.norm(W36[:, r_remove] - W_rem @ coeff_W)
            else:
                W_err = np.linalg.norm(W36[:, r_remove])
            
            # If all errors are very small, we might be able to merge
            if U_err < 1e-6 and V_err < 1e-6 and W_err < 1e-6:
                # Try to construct the rank-35 solution
                # The idea: U_new[:, r_remove] is not present, but the contribution
                # of the removed term is redistributed.
                #
                # Actually, this approach is flawed. The tensor decomposition is not
                # simply about linear dependencies in each factor separately.
                # The trilinear form sum_r U[i,r]*V[j,r]*W[k,r] = 1 must be preserved.
                #
                # Let's try a different approach: directly optimize with the constraint
                # that one rank term is zero.
                
                pass
        
        except Exception:
            continue
    
    # Try direct optimization for rank 35 with multiple restarts
    target_T = _build_tensor(2, 4, 5)
    result = _optimize_rank35(target_T, 8, 20, 10, deadline)
    if result is not None:
        return result
    
    return None

def _build_schoolbook():
    """Build the schoolbook rank-40 decomposition."""
    n, m, p = 2, 4, 5
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p  # 40
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    
    n, m, p = 2, 4, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook()
    best_result = {
        "U": [[float(U_school[i, r]) for r in range(U_school.shape[1])] for i in range(nm)],
        "V": [[float(V_school[i, r]) for r in range(V_school.shape[1])] for i in range(mp)],
        "W": [[float(W_school[i, r]) for r in range(W_school.shape[1])] for i in range(pn)]
    }
    best_rank = 40
    
    # Try rank 36 Strassen decomposition
    U36, V36, W36 = _build_strassen_rank36()
    if _verify_decomposition(U36, V36, W36, n, m, p):
        best_result = {
            "U": [[float(U36[i, r]) for r in range(36)] for i in range(nm)],
            "V": [[float(V36[i, r]) for r in range(36)] for i in range(mp)],
            "W": [[float(W36[i, r]) for r in range(36)] for i in range(pn)]
        }
        best_rank = 36
    
    # Try to find rank 35
    if time.time() < deadline - 30:
        result = _try_groebner_rank35(deadline)
        if result is not None:
            U35, V35, W35, r = result
            if _verify_decomposition(U35, V35, W35, n, m, p):
                best_result = {
                    "U": [[float(U35[i, r2]) for r2 in range(r)] for i in range(nm)],
                    "V": [[float(V35[i, r2]) for r2 in range(r)] for i in range(mp)],
                    "W": [[float(W35[i, r2]) for r2 in range(r)] for i in range(pn)]
                }
                best_rank = r
    
    # Round to half-integers for cleanliness
    for key in ["U", "V", "W"]:
        best_result[key] = [[int(round(val * 2)) / 2 for val in row] for row in best_result[key]]
    
    return best_result

# EVOLVE-BLOCK-END
