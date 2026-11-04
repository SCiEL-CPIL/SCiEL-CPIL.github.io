"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the (n*m) x (m*p) x (p*n) tensor T where T[a,b,c]=1 iff consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return bool(np.allclose(result, T, atol=tol))

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition (always valid)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = n*m*p
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _rank17(n, m, p):
    """Verified Strassen(2x2x2) + direct decomposition of rank 17."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = 17
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))

    def set_term(r, u, v, w):
        for a, val in u.items(): U[a, r] = val
        for b, val in v.items(): V[b, r] = val
        for c, val in w.items(): W[c, r] = val

    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2,0:2]
    set_term(0, {0:1,4:1}, {0:1,4:1}, {0:1,3:1})   # M1=(a0+a4)(b0+b4)
    set_term(1, {3:1,4:1}, {0:1},       {1:1,3:-1}) # M2=(a3+a4)b0
    set_term(2, {0:1},       {1:1,4:-1}, {2:1,3:1}) # M3=a0(b1-b4)
    set_term(3, {4:1},       {3:1,0:-1}, {0:1,1:1}) # M4=a4(b3-b0)
    set_term(4, {0:1,1:1},   {4:1},      {0:-1,2:1})# M5=(a0+a1)b4
    set_term(5, {3:1,0:-1},  {0:1,1:1},  {3:1})     # M6=(a3-a0)(b0+b1)
    set_term(6, {1:1,4:-1},  {3:1,4:1},  {0:1})     # M7=(a1-a4)(b3+b4)
    # Part 2: A1(2x2) x B1_right(2x1) -> C[:,2]
    set_term(7,  {0:1}, {2:1}, {4:1})
    set_term(8,  {1:1}, {5:1}, {4:1})
    set_term(9,  {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    # Part 3: A2(2x1) x B2(1x3) -> C[0:2,0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    return U, V, W

def _try_reduced(T, U, V, W, R, deadline):
    """
    Bounded search for a strictly smaller rank decomposition.
    Only returns a solution if it is EXACTLY verified; otherwise returns None.
    """
    n_a, n_b, n_c = T.shape
    rank = U.shape[1]
    best = None

    # (a) single-term drop: try removing each term
    for drop in range(rank):
        if time.time() > deadline: break
        keep = [t for t in range(rank) if t != drop]
        Ux = np.zeros((n_a, len(keep))); Vx = np.zeros((n_b, len(keep))); Wx = np.zeros((n_c, len(keep)))
        for ci, ti in enumerate(keep):
            Ux[:, ci] = U[:, ti]; Vx[:, ci] = V[:, ti]; Wx[:, ci] = W[:, ti]
        if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
            best = (Ux, Vx, Wx)
            break

    if best is not None:
        return best

    # (b) drop two terms, add one new rank-1 term from a small structured grid.
    grid = [-2, -1, 0, 1, 2]
    for d1 in range(rank):
        if time.time() > deadline: break
        for d2 in range(d1+1, rank):
            if time.time() > deadline: break
            keep = [t for t in range(rank) if t not in (d1, d2)]
            for ca in grid:
                for cb in grid:
                    if ca == 0 and cb == 0:
                        continue
                    uu = ca*U[:, d1] + cb*U[:, d2]
                    vv = ca*V[:, d1] + cb*V[:, d2]
                    ww = ca*W[:, d1] + cb*W[:, d2]
                    Ux = np.zeros((n_a, len(keep)+1)); Vx = np.zeros((n_b, len(keep)+1)); Wx = np.zeros((n_c, len(keep)+1))
                    for ci, ti in enumerate(keep):
                        Ux[:, ci] = U[:, ti]; Vx[:, ci] = V[:, ti]; Wx[:, ci] = W[:, ti]
                    Ux[:, len(keep)] = uu; Vx[:, len(keep)] = vv; Wx[:, len(keep)] = ww
                    if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
                        return (Ux, Vx, Wx)
    
    # (c) Randomized search for rank 16 using aligned sparse terms
    # We generate candidate terms that are non-zero only on "aligned" indices.
    # An aligned term for a set of (i,j,k) tuples has:
    # u non-zero at i*m+j, v at j*p+k, w at k*n+i.
    # We will try to find 16 such terms that sum to T.
    
    np.random.seed(42)
    
    # Pre-compute all possible "aligned" rank-1 tensors with small integer coefficients.
    # To limit the pool, we only consider terms that are non-zero on a small number of components.
    # We will generate random supports.
    
    # For n=2, m=3, p=3:
    # a in 0..5, b in 0..8, c in 0..5.
    # T[a,b,c] = 1 if a=i*3+j, b=j*3+k, c=k*2+i.
    
    # We will try to find a rank 16 decomposition by solving for coefficients of 16 random candidate terms.
    # If the system is consistent, we have a solution.
    
    max_restarts = 500
    for _ in range(max_restarts):
        if time.time() > deadline: break
        
        # Generate 16 random candidate terms
        # Each term is a sparse vector in R^6, R^9, R^6.
        # We bias the generation towards indices that are part of T's support.
        
        U_cand = np.zeros((n_a, 16))
        V_cand = np.zeros((n_b, 16))
        W_cand = np.zeros((n_c, 16))
        
        for r in range(16):
            # Pick a random subset of components (i,j,k) to cover
            # For simplicity, pick 1-3 components
            num_comps = np.random.randint(1, 4)
            comps = np.random.choice(12, num_comps, replace=False)
            
            u_vec = np.zeros(n_a)
            v_vec = np.zeros(n_b)
            w_vec = np.zeros(n_c)
            
            for comp in comps:
                i = comp // 9
                j = (comp % 9) // 3
                k = comp % 3
                
                a = i*3 + j
                b = j*3 + k
                c = k*2 + i
                
                # Assign random coefficients from {-1, 1}
                u_vec[a] += np.random.choice([-1, 1])
                v_vec[b] += np.random.choice([-1, 1])
                w_vec[c] += np.random.choice([-1, 1])
            
            U_cand[:, r] = u_vec
            V_cand[:, r] = v_vec
            W_cand[:, r] = w_vec
            
        # Now we want to find coefficients c_r such that sum c_r * U_cand[:,r] V_cand[:,r] W_cand[:,r] == T
        # This is a linear system in c_r.
        # Flatten T and the outer products.
        
        T_flat = T.flatten()
        # Construct matrix A where A[:, r] is the flattened outer product of U_cand[:,r], V_cand[:,r], W_cand[:,r]
        # This is expensive for large dimensions, but 6*9*6 = 324 is small.
        
        A = np.zeros((324, 16))
        for r in range(16):
            outer = np.einsum('a,b,c->abc', U_cand[:, r], V_cand[:, r], W_cand[:, r])
            A[:, r] = outer.flatten()
            
        # Solve A c = T_flat
        # Use least squares, then check if the residual is zero.
        try:
            c, residuals, rank_s, s = np.linalg.lstsq(A, T_flat, rcond=None)
            # Check if the solution is exact
            if np.allclose(A @ c, T_flat, atol=1e-8):
                # Verify the decomposition
                U_sol = U_cand @ np.diag(c) # This is not right, U_sol should be U_cand with columns scaled by c?
                # No, the decomposition is sum c_r (u_r v_r w_r) = sum (c_r u_r) v_r w_r?
                # Or (c_r^{1/3} u_r) ... No, we can absorb c into any of the vectors.
                # Let U_sol[:, r] = c_r * U_cand[:, r]
                U_sol = U_cand.copy()
                for r in range(16):
                    U_sol[:, r] *= c[r]
                # Check if entries are close to integers/half-integers?
                # The problem allows integer/half-integer entries.
                # If c are rational, we can scale.
                
                if _verify_decomposition(U_sol, V_cand, W_cand, T):
                    return (U_sol, V_cand, W_cand)
        except np.linalg.LinAlgError:
            pass
            
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # explicit safety margin (< 120s wall-clock)

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Baseline: verified rank-17 (Strassen + direct).
    U17, V17, W17 = _rank17(n, m, p)
    if _verify_decomposition(U17, V17, W17, T):
        base = (U17, V17, W17)
    else:
        base = _schoolbook(n, m, p)

    # Bounded refinement: attempt a strictly smaller, EXACTLY verified rank.
    # This never hurts: it only replaces the baseline if verified.
    reduced = _try_reduced(T, base[0], base[1], base[2], base[0].shape[1], deadline)
    if reduced is not None:
        U, V, W = reduced
    else:
        U, V, W = base

    # Final exact verification; fall back to schoolbook on any failure.
    if not _verify_decomposition(U, V, W, T):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
