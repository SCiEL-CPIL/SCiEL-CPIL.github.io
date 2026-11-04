import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def reconstruct(U, V, W, rank):
        """Compute Rec[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r, optimize=True)
        return Rec

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = reconstruct(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def error(U, V, W, rank):
        Rec = reconstruct(U, V, W, rank)
        return np.sum((Rec - 1.0) ** 2)

    def als_solve(U, V, W, rank, max_iters=200, tol=1e-14, lambda_reg=1e-10):
        """Alternating Least Squares to minimize sum (Rec - 1)^2.
        
        For each r, the contribution to Rec[i,j,k] is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
        We alternate: fix V,W solve for U; fix U,W solve for V; fix U,V solve for W.
        
        For U[:,r]: given V[:,r] and W[:,r], we want to find u such that
        sum_i,j,k (sum_r' u_r'[i*m+j]*v_r'[j*p+k]*w_r'[k*n+i] - 1)^2 is minimized w.r.t u_r.
        
        This is a linear system. Let's think of it differently.
        
        Actually, the tensor T[i,j,k] = 1. We want T = sum_r u_r ⊗ v_r ⊗ w_r where
        u_r[i,j] = U[i*m+j,r], v_r[j,k] = V[j*p+k,r], w_r[k,i] = W[k*n+i,r].
        
        The constraint is: for all i,j,k: sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1
        
        ALS for U: fix V,W. For each r, we want to minimize
        ||T - sum_r' u_r' ⊙ v_r' ⊙ w_r'||^2 where ⊙ means the "cyclic" product.
        
        Let R = T - sum_{r'≠r} u_r'[i,j]*v_r'[j*p+k]*w_r'[k*n+i]
        Then we want to minimize ||R - u_r[i,j]*v_r[j*p+k]*w_r[k*n+i]||^2
        
        This is: min_u ||R - u ⊙ v_r ⊙ w_r||^2 where (u⊙v⊙w)[i,j,k] = u[i,j]*v[j,k]*w[k,i]
        
        This is a linear least squares problem in u!
        For each (i,j), the coefficient of u[i,j] in the objective is:
        sum_{k} v_r[j,k]^2 * w_r[k,i]^2
        
        And the cross term with R[i,j,k] is:
        sum_k R[i,j,k] * v_r[j,k] * w_r[k,i]
        
        So we solve: for each (i,j):
        (sum_k v_r[j,k]^2 * w_r[k,i]^2) * u[i,j] = sum_k R[i,j,k] * v_r[j,k] * w_r[k,i]
        
        Similarly for V and W.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, it
            
            old_err = error(U, V, W, rank)
            
            # Update U
            for r in range(rank):
                # Compute residual without this term
                # R[i,j,k] = 1 - sum_{r'≠r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r']
                # = 1 - (sum_r' U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] - U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])
                # Let's compute full sum first
                full_sum = np.zeros((n, m, p))
                for r2 in range(rank):
                    u2 = U[:, r2].reshape(n, m)
                    v2 = V[:, r2].reshape(m, p)
                    w2 = W[:, r2].reshape(p, n)
                    full_sum += np.einsum('ij,jk,ki->ijk', u2, v2, w2, optimize=True)
                
                R = 1.0 - (full_sum - U[:, r].reshape(n, m) * V[:, r].reshape(m, p)[:, :, None] * W[:, r].reshape(p, n)[None, :, :])
                # Actually simpler: R = 1 - full_sum + u_r⊙v_r⊙w_r
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                R = 1.0 - full_sum + np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r, optimize=True)
                
                # Solve for u_r: for each (i,j):
                # (sum_k v_r[j,k]^2 * w_r[k,i]^2) * u_new[i,j] = sum_k R[i,j,k] * v_r[j,k] * w_r[k,i]
                for i in range(n):
                    for j in range(m):
                        denom = np.sum(v_r[j, :] ** 2 * w_r[:, i] ** 2)
                        numer = np.sum(R[i, j, :] * v_r[j, :] * w_r[:, i])
                        if denom > lambda_reg:
                            U[i * m + j, r] = numer / denom
                        else:
                            U[i * m + j, r] = 0.0
            
            # Update V
            for r in range(rank):
                full_sum = np.zeros((n, m, p))
                for r2 in range(rank):
                    u2 = U[:, r2].reshape(n, m)
                    v2 = V[:, r2].reshape(m, p)
                    w2 = W[:, r2].reshape(p, n)
                    full_sum += np.einsum('ij,jk,ki->ijk', u2, v2, w2, optimize=True)
                
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                R = 1.0 - full_sum + np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r, optimize=True)
                
                # For each (j,k):
                # (sum_i u_r[i,j]^2 * w_r[k,i]^2) * v_new[j,k] = sum_i R[i,j,k] * u_r[i,j] * w_r[k,i]
                for j in range(m):
                    for k in range(p):
                        denom = np.sum(u_r[:, j] ** 2 * w_r[k, :] ** 2)
                        numer = np.sum(R[:, j, k] * u_r[:, j] * w_r[k, :])
                        if denom > lambda_reg:
                            V[j * p + k, r] = numer / denom
                        else:
                            V[j * p + k, r] = 0.0
            
            # Update W
            for r in range(rank):
                full_sum = np.zeros((n, m, p))
                for r2 in range(rank):
                    u2 = U[:, r2].reshape(n, m)
                    v2 = V[:, r2].reshape(m, p)
                    w2 = W[:, r2].reshape(p, n)
                    full_sum += np.einsum('ij,jk,ki->ijk', u2, v2, w2, optimize=True)
                
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                R = 1.0 - full_sum + np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r, optimize=True)
                
                # For each (k,i):
                # (sum_j u_r[i,j]^2 * v_r[j,k]^2) * w_new[k,i] = sum_j R[i,j,k] * u_r[i,j] * v_r[j,k]
                for k in range(p):
                    for i in range(n):
                        denom = np.sum(u_r[i, :] ** 2 * v_r[:, k] ** 2)
                        numer = np.sum(R[i, :, k] * u_r[i, :] * v_r[:, k])
                        if denom > lambda_reg:
                            W[k * n + i, r] = numer / denom
                        else:
                            W[k * n + i, r] = 0.0
            
            new_err = error(U, V, W, rank)
            if abs(old_err - new_err) < tol:
                break
        
        return U, V, W, it + 1

    def try_rank_als(rank, num_restarts=5, max_iters=100):
        nonlocal best_solution, best_rank
        if rank >= best_rank:
            return False
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return False
            
            # Initialize: start from schoolbook and randomly zero out some columns
            U, V, W = random_init_from_schoolbook(rank)
            
            # Run ALS
            U, V, W, iters = als_solve(U, V, W, rank, max_iters=max_iters)
            
            if is_valid(U, V, W, rank, atol=1e-9):
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = rank
                return True
            
            # Try snapping to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if is_valid(U_snap, V_snap, W_snap, rank, atol=1e-9):
                best_solution = {"U": U_snap, "V": V_snap, "W": W_snap}
                best_rank = rank
                return True
            
            # Try integer snapping
            U_snap_i = np.round(U)
            V_snap_i = np.round(V)
            W_snap_i = np.round(W)
            if is_valid(U_snap_i, V_snap_i, W_snap_i, rank, atol=1e-9):
                best_solution = {"U": U_snap_i, "V": V_snap_i, "W": W_snap_i}
                best_rank = rank
                return True
        
        return False

    def random_init_from_schoolbook(rank):
        """Initialize by taking a subset of schoolbook columns and adding small noise."""
        n_schoolbook = n * m * p
        # Pick 'rank' random columns from schoolbook
        indices = np.random.choice(n_schoolbook, size=rank, replace=False)
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for new_col, old_col in enumerate(indices):
            i = old_col // (m * p)
            j = (old_col // p) % m
            k = old_col % p
            U[i * m + j, new_col] = 1.0
            V[j * p + k, new_col] = 1.0
            W[k * n + i, new_col] = 1.0
        
        # Add small random perturbation to break symmetry
        U += np.random.randn(n * m, rank) * 0.1
        V += np.random.randn(m * p, rank) * 0.1
        W += np.random.randn(p * n, rank) * 0.1
        
        return U, V, W

    def random_init_pure(rank):
        """Pure random initialization."""
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        return U, V, W

    # Search from best_rank - 1 downwards
    current_rank = best_rank - 1
    while current_rank >= max(15, best_rank // 2) and time.time() < deadline:
        # Try ALS with different initializations
        found = try_rank_als(current_rank, num_restarts=3, max_iters=80)
        if found:
            break
        
        # Also try pure random init with ALS
        for restart in range(2):
            if time.time() > deadline:
                break
            U, V, W = random_init_pure(current_rank)
            U, V, W, iters = als_solve(U, V, W, current_rank, max_iters=80)
            if is_valid(U, V, W, current_rank, atol=1e-9):
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = current_rank
                break
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if is_valid(U_snap, V_snap, W_snap, current_rank, atol=1e-9):
                best_solution = {"U": U_snap, "V": V_snap, "W": W_snap}
                best_rank = current_rank
                break
        
        if best_rank == current_rank:
            break
        
        current_rank -= 1

    # Ensure the returned values are lists of lists of floats (or ints)
    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
