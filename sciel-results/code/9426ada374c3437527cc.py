import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # First try to find rank-16 via structured algebraic search
    result_16 = _try_rank16_algebraic(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17
    return _rank17_decomposition(n, m, p)


def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition with half-integer rounding."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    Ui = np.rint(U_s * 2).astype(np.int64)
    Vi = np.rint(V_s * 2).astype(np.int64)
    Wi = np.rint(W_s * 2).astype(np.int64)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None


def _try_rank16_algebraic(n, m, p, deadline):
    """
    Structured algebraic search for rank-16 decomposition.
    
    Strategy: Fix V to have rank-1 structure per term (V[j,k,r] = v_j[r]*w_k[r]),
    then solve the resulting linear system for W given fixed U and V.
    This exploits the all-ones target tensor structure.
    """
    rank = 16
    target = np.ones((n, m, p))
    
    np.random.seed(42)
    
    # Strategy 1: Fix V to rank-1 structure per term, randomize U, solve for W
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Generate V with rank-1 structure: V[j,k,r] = v_j[r] * w_k[r]
        # v and w are vectors of length m and p respectively, for each r
        V = np.zeros((m * p, rank))
        for r in range(rank):
            v_vec = np.zeros(m)
            w_vec = np.zeros(p)
            # Choose 1-2 non-zero entries in v_vec
            nz_v = np.random.choice(m, size=np.random.randint(1, 3), replace=False)
            for j in nz_v:
                v_vec[j] = np.random.choice([-1, 1])
            # Choose 1-2 non-zero entries in w_vec
            nz_w = np.random.choice(p, size=np.random.randint(1, 3), replace=False)
            for k in nz_w:
                w_vec[k] = np.random.choice([-1, 1])
            for j in range(m):
                for k in range(p):
                    V[j * p + k, r] = v_vec[j] * w_vec[k]
        
        # Generate sparse U
        U = np.zeros((n * m, rank))
        for r in range(rank):
            nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1, 1])
        
        # Build linear system for W:
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let W_flat[(k,i,r)] = W[k*n+i, r]
        # Coefficient of W_flat[(k,i,r)] in equation (i,j,k) is U[i*m+j, r] * V[j*p+k, r]
        
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        col_idx = (k * n + i) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
        
        # Solve least squares
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            W_sol = W_flat.reshape(p * n, rank)
        except:
            continue
        
        # Check if solution is close to half-integers
        W_rounded = np.round(W_sol * 2) / 2.0
        W_int = np.rint(W_rounded * 2).astype(np.int64)
        
        # Verify with rounded values
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        Ui = np.rint(U_r * 2).astype(np.int64).reshape(n, m, rank)
        Vi = np.rint(V_r * 2).astype(np.int64).reshape(m, p, rank)
        Wi = np.rint(W_rounded * 2).astype(np.int64).reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
        
        if np.all(prod == 8):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_rounded.tolist()
            }
        
        # Also try: keep W as solved (not rounded), round only U and V
        # This won't work for exact verification, but check residual
        err = np.sum((prod - 8)**2)
    
    # Strategy 2: Fix U to rank-1 structure, randomize V, solve for W
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Generate U with rank-1 structure: U[i,j,r] = u_i[r] * v_j[r]
        U = np.zeros((n * m, rank))
        for r in range(rank):
            u_vec = np.zeros(n)
            v_vec = np.zeros(m)
            nz_u = np.random.choice(n, size=np.random.randint(1, 2), replace=False)
            for i in nz_u:
                u_vec[i] = np.random.choice([-1, 1])
            nz_v = np.random.choice(m, size=np.random.randint(1, 3), replace=False)
            for j in nz_v:
                v_vec[j] = np.random.choice([-1, 1])
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = u_vec[i] * v_vec[j]
        
        # Generate sparse V
        V = np.zeros((m * p, rank))
        for r in range(rank):
            nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Build linear system for W
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        col_idx = (k * n + i) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
        
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            W_sol = W_flat.reshape(p * n, rank)
        except:
            continue
        
        W_rounded = np.round(W_sol * 2) / 2.0
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        Ui = np.rint(U_r * 2).astype(np.int64).reshape(n, m, rank)
        Vi = np.rint(V_r * 2).astype(np.int64).reshape(m, p, rank)
        Wi = np.rint(W_rounded * 2).astype(np.int64).reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
        
        if np.all(prod == 8):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_rounded.tolist()
            }
    
    # Strategy 3: Fix W to have one non-zero per (k,i) pair, solve for U and V jointly
    # For each (k,i), assign to a unique r, then W[k,i,r]=1
    # This makes the problem: for each (k,i), sum_j U[i,j,r_ki]*V[j,k,r_ki] = 1
    # Enumerate assignments of 6 (k,i) pairs to 16 terms
    
    from itertools import product
    
    # Try all assignments where each (k,i) gets a distinct r
    # There are 16*15*14*13*12*11 possible assignments - too many
    # Instead, try structured assignments: group by k or by i
    
    # Assignment type A: all (k,0) go to r=0,1,2; all (k,1) go to r=3,4,5
    # Then for k=0: sum_j U[0,j,0]*V[j,0,0] = 1, sum_j U[1,j,3]*V[j,0,3] = 1
    # etc. This gives 6 independent 3-variable linear systems
    
    for assign_type in range(4):
        if time.time() > deadline:
            break
        
        # Define assignment of (k,i) to r
        r_assign = {}
        if assign_type == 0:
            # (k,0) -> k, (k,1) -> k+3
            for k in range(p):
                r_assign[(k, 0)] = k
                r_assign[(k, 1)] = k + 3
        elif assign_type == 1:
            # (k,0) -> 2*k, (k,1) -> 2*k+1
            for k in range(p):
                r_assign[(k, 0)] = 2 * k
                r_assign[(k, 1)] = 2 * k + 1
        elif assign_type == 2:
            # (0,i) -> i, (1,i) -> i+2, (2,i) -> i+4
            for i in range(n):
                for k in range(p):
                    r_assign[(k, i)] = i * p + k
        else:
            # Random assignment
            r_assign = {}
            used = set()
            for k in range(p):
                for i in range(n):
                    r = np.random.randint(0, rank)
                    while r in used:
                        r = np.random.randint(0, rank)
                    r_assign[(k, i)] = r
                    used.add(r)
        
        # Build W from assignment
        W = np.zeros((p * n, rank))
        for (k, i), r in r_assign.items():
            W[k * n + i, r] = 1
        
        # Now for each (k,i), we need: sum_j U[i,j,r_ki]*V[j,k,r_ki] = 1
        # Let r_ki = r_assign[(k,i)]
        # This is: for each (k,i), a linear system in U[i,*,r_ki] and V[*,k,r_ki]
        
        # Collect all U[i,j,r] and V[j,k,r] variables that appear
        # U variables: U[i,j,r] where r = r_assign[(k,i)] for some k
        # V variables: V[j,k,r] where r = r_assign[(k,i)] for some i
        
        # Let's build a joint linear system
        # Variables: all U[i*m+j, r] and V[j*p+k, r] that are used
        # Equations: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Since W[k*n+i,r] is 1 only when r = r_assign[(k,i)], we have:
        # U[i*m+j, r_assign[(k,i)]] * V[j*p+k, r_assign[(k,i)]] = 1
        
        # This is a system of bilinear equations. Let's linearize by fixing U or V.
        
        # Fix U to random values, solve for V
        U = np.zeros((n * m, rank))
        for r in range(rank):
            nz = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz:
                U[idx, r] = np.random.choice([-1, 1])
        
        # For each (k,i), r_ki = r_assign[(k,i)]
        # Need: sum_j U[i*m+j, r_ki] * V[j*p+k, r_ki] = 1
        # This is a linear system in V[*,k,r_ki] for each (k,i)
        
        V = np.zeros((m * p, rank))
        valid = True
        for k in range(p):
            for i in range(n):
                r_ki = r_assign[(k, i)]
                # Equation: sum_j U[i*m+j, r_ki] * V[j*p+k, r_ki] = 1
                # This is m equations... wait, no, it's one equation with m unknowns
                # Actually it's: U[i,0,r_ki]*V[0,k,r_ki] + U[i,1,r_ki]*V[1,k,r_ki] + U[i,2,r_ki]*V[2,k,r_ki] = 1
                # This is one equation in 3 unknowns - underdetermined
                
                # Let's fix 2 of the 3 V values randomly, solve for the 3rd
                # Or better: fix all U values, then for each (k,i), we have one equation
                # in 3 V variables. We need to ensure consistency across i for the same k.
                
                # For fixed k, we have n equations (one per i) in m*p... no, in m variables V[*,k,r_ki]
                # But r_ki depends on i, so different i use different r
                # This means the V variables are actually independent across i for fixed k
                # (since they have different r indices)
                
                # So for each (k,i), we have one equation in m V variables:
                # sum_j U[i*m+j, r_ki] * V[j*p+k, r_ki] = 1
                # We can solve this by setting m-1 variables to 0 and solving for the 1 remaining
                # But we need U[i*m+j, r_ki] != 0 for at least one j
                
                # Choose j* such that U[i*m+j*, r_ki] != 0
                found = False
                for j_star in range(m):
                    if U[i * m + j_star, r_ki] != 0:
                        # Set V[j*p+k, r_ki] = 0 for all j != j_star
                        # V[j_star*p+k, r_ki] = 1 / U[i*m+j_star, r_ki]
                        for j in range(m):
                            if j != j_star:
                                V[j * p + k, r_ki] = 0
                        V[j_star * p + k, r_ki] = 1.0 / U[i * m + j_star, r_ki]
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        Ui = np.rint(U_r * 2).astype(np.int64).reshape(n, m, rank)
        Vi = np.rint(V_r * 2).astype(np.int64).reshape(m, p, rank)
        Wi = np.rint(W_r * 2).astype(np.int64).reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
        
        if np.all(prod == 8):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
    
    return None


def _newton_refine(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Newton's method refinement for tensor decomposition."""
    rank = U.shape[1]
    target = np.ones((n, m, p))
    
    for it in range(max_iter):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - target
        err = np.sum(res**2)
        
        if err < tol:
            break
        
        # Compute gradients (Jacobian of the residual)
        # dU[i,j,r] = sum_{k} res[i,j,k] * V[j,k,r] * W[k,i,r]
        dU = np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        # Approximate Hessian using Gauss-Newton (J^T J)
        # For speed, use a damped Newton step with adaptive step size
        dU_flat = dU.reshape(-1)
        dV_flat = dV.reshape(-1)
        dW_flat = dW.reshape(-1)
        
        # Gradient norm
        grad_norm = np.sqrt(np.sum(dU_flat**2) + np.sum(dV_flat**2) + np.sum(dW_flat**2))
        
        if grad_norm < 1e-15:
            break
        
        # Adaptive step size with backtracking
        step = 1.0
        for _ in range(20):
            U_new = U - step * dU_flat.reshape(U.shape)
            V_new = V - step * dV_flat.reshape(V.shape)
            W_new = W - step * dW_flat.reshape(W.shape)
            
            U_t_new = U_new.reshape(n, m, rank)
            V_t_new = V_new.reshape(m, p, rank)
            W_t_new = W_new.reshape(p, n, rank)
            prod_new = np.einsum('ijr,jkr,kir->ijk', U_t_new, V_t_new, W_t_new)
            err_new = np.sum((prod_new - target)**2)
            
            if err_new < err:
                U, V, W = U_new, V_new, W_new
                break
            step *= 0.5
        
        else:
            # No improvement, try smaller step or break
            break
    
    return U, V, W


def _try_rank16_direct(n, m, p, deadline):
    """
    Direct structured search: exploit the fact that the target tensor is all-ones.
    
    Key observation: T[i,j,k] = 1 for all (i,j,k). This means the decomposition
    must satisfy: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    We can fix W to have a specific structure and solve for U and V.
    """
    rank = 16
    target = np.ones((n, m, p))
    
    # Fix W to a structured pattern: W[k,i,r] = 1 for specific (k,i,r)
    # Try: W has at most one non-zero per (k,i) pair across all r
    # i.e., for each (k,i), there's exactly one r where W[k,i,r] != 0
    
    # This means: for each (k,i), the contribution to T[i,j,k] from all r is
    # sum_j' U[i,j',r_ki]*V[j',k,r_ki] where r_ki is the unique r for (k,i)
    # This must equal 1 for all j.
    
    # So we need: for each (k,i), there exists r such that
    # sum_j U[i,j,r]*V[j,k,r] = 1 (with W[k,i,r]=1)
    
    # This is a system of linear equations in U and V!
    
    # Let's enumerate possible assignments of r to (k,i) pairs
    # There are p*n = 6 (k,i) pairs and rank=16 terms
    # Each (k,i) can be assigned to any of the 16 terms
    
    # But this is a huge search space (16^6). Let's be smarter.
    
    # Alternative: Fix U and V to have simple structure, solve for W
    # If U[i,j,r] and V[j,k,r] are fixed, then for each (i,j,k):
    # W[k,i,r] must satisfy: sum_r W[k,i,r] * (U[i,j,r]*V[j,k,r]) = 1
    # This is a linear system in W!
    
    # Let's try: fix U and V to sparse patterns, solve linear system for W
    
    np.random.seed(123)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate sparse U and V
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        for r in range(rank):
            # U[i,j,r]: pick 1-2 (i,j) pairs
            nz = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz:
                U[idx, r] = np.random.choice([-1, 1])
            
            nz = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Now solve for W: for each (i,j,k), sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
        # This is: for each (i,j,k): A_{(i,j,k), (k',i')} ... hmm, this is not linear in W alone
        # because the coefficient depends on (i,j,k)
        
        # Actually, let me reconsider. For fixed U and V:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Let's define B[i,j,k,r] = U[i,j,r] * V[j,k,r]
        # Then T[i,j,k] = sum_r B[i,j,k,r] * W[k,i,r]
        # 
        # For fixed (i,k), varying j: T[i,j,k] = sum_r B[i,j,k,r] * W[k,i,r] = 1
        # This gives m equations in the p*n variables W[k,i,r] for r=0..rank-1
        # But W[k,i,r] appears in multiple (i,j,k) equations
        # 
        # Total equations: n*m*p = 18
        # Total unknowns: p*n*rank = 6*16 = 96
        # Underdetermined! We can solve this as a least-squares problem.
        
        # Build the linear system: for each (i,j,k), equation:
        # sum_r B[i,j,k,r] * W[k,i,r] = 1
        # 
        # Let's index W as W_flat[(k,i,r)] = W[k*n+i, r]
        # For equation (i,j,k): coefficient of W_flat[(k,i,r)] is B[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r]
        
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        col_idx = (k * n + i) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
        
        # Solve least squares
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            W_sol = W_flat.reshape(p * n, rank)
        except:
            continue
        
        # Check if solution is close to half-integers
        W_rounded = np.round(W_sol * 2) / 2.0
        W_int = np.rint(W_rounded * 2).astype(np.int64)
        
        # Verify
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W_int.reshape(p, n, rank)
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        if np.all(prod == 8):
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = W_rounded
            return {
                "U": U_s.tolist(),
                "V": V_s.tolist(),
                "W": W_s.tolist()
            }
        
        # Also try without rounding W, just check if U and V are already half-integers
        # and if the residual is small
        err = np.sum((prod - 8)**2)
        if err < 1e-6:
            # Try rounding all
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W_sol * 2) / 2.0
            Ui = np.rint(U_r * 2).astype(np.int64).reshape(n, m, rank)
            Vi = np.rint(V_r * 2).astype(np.int64).reshape(m, p, rank)
            Wi = np.rint(W_r * 2).astype(np.int64).reshape(p, n, rank)
            prod2 = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
            if np.all(prod2 == 8):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    return None
