"""
Matrix multiplication 3x3x4 - rank-minimal tensor decomposition.
Exploits the block structure to reduce the search space significantly.
"""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 138  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    N, M, P = n * m, m * p, p * n  # 9, 12, 12
    
    # Build the target tensor T[N, M, P]
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute which (a,b,c) triples are nonzero
    nz_indices = np.argwhere(T == 1.0)
    nz_count = len(nz_indices)  # 36
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U@V@W reconstructs T exactly."""
        # Compute T_reconstructed = sum_r U[:,r] * V[:,r] * W[:,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)
    
    def error(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sqrt(np.sum((T_rec - T) ** 2))
    
    # Strategy: exploit the block structure.
    # For each column r, U[:,r], V[:,r], W[:,r] define a rank-1 slice.
    # The key insight: T has a very specific sparse structure.
    # Each nonzero entry T[a,b,c]=1 corresponds to a unique (i,j,k) triple.
    # 
    # The tensor can be seen as: T[a,b,c] = 1 iff a%m == b//p, a//m == c%n, b%p == c//n
    # i.e., the "row index" of A, the "col index" of B (which is row of C), and "col index" of C
    # are all consistently linked through the intermediate index j.
    #
    # This is the standard matrix multiplication tensor. For 3x3x4, the known
    # lower bound is 18 (since rank >= max(nm, mp, pn) is not quite right, but
    # rank >= 2 * max(n,m,p) - 1 for square, and for rectangular we have bounds).
    # Actually, for 3x3x4, the rank is known to be at most 24 (Laderman-like) 
    # and at least 18.
    
    # Use structured initialization and alternating least squares.
    # Key optimization: for each target rank, use the fact that we only need
    # to match 36 specific entries (the 1s), not all N*M*P entries.
    
    def solve_W(U, V, r):
        """Given U, V of shape [*, r], find W of shape [P, r] minimizing error."""
        # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
        # For each c: sum_{a,b} T[a,b,c] * ... 
        # Actually, for fixed c: sum_r (U[:,r] * V[:,r])^T * W[c,r] = T[:,:,c].flatten()
        # So for each c, we have a linear system in W[c, :]
        W_out = np.zeros((P, r))
        for c in range(P):
            rhs = T[:, :, c].flatten()  # shape (N*M,)
            # A is (N*M) x r, where A[(a,b), r] = U[a,r]*V[b,r]
            A = np.outer(U.flatten(), V.flatten()).T.reshape(N*M, r) if False else None
            # More efficiently:
            A = np.zeros((N * M, r))
            idx = 0
            for a in range(N):
                for b in range(M):
                    A[idx, :] = U[a, :] * V[b, :]
                    idx += 1
            # Solve A @ W[c,:] = rhs
            W_c, _, _, _ = np.linalg.lstsq(A, rhs, rcond=1e-12)
            W_out[c, :] = W_c
        return W_out
    
    def solve_U(V, W, r):
        """Given V, W of shape [*, r], find U of shape [N, r] minimizing error."""
        U_out = np.zeros((N, r))
        for a in range(N):
            rhs = T[a, :, :].flatten()  # shape (M*P,)
            A = np.zeros((M * P, r))
            idx = 0
            for b in range(M):
                for c in range(P):
                    A[idx, :] = V[b, :] * W[c, :]
                    idx += 1
            U_a, _, _, _ = np.linalg.lstsq(A, rhs, rcond=1e-12)
            U_out[a, :] = U_a
        return U_out
    
    def solve_V(U, W, r):
        """Given U, W of shape [*, r], find V of shape [M, r] minimizing error."""
        V_out = np.zeros((M, r))
        for b in range(M):
            rhs = T[:, b, :].flatten()  # shape (N*P,)
            A = np.zeros((N * P, r))
            idx = 0
            for a in range(N):
                for c in range(P):
                    A[idx, :] = U[a, :] * W[c, :]
                    idx += 1
            V_b, _, _, _ = np.linalg.lstsq(A, rhs, rcond=1e-12)
            V_out[b, :] = V_b
        return V_out
    
    # Fallback: schoolbook rank-36
    rank_school = N * M  # no, it's n*m*p = 36
    rank_school = n * m * p
    U_sch = np.zeros((N, rank_school))
    V_sch = np.zeros((M, rank_school))
    W_sch = np.zeros((P, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1
                V_sch[j * p + k, col] = 1
                W_sch[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    # Try ranks from 18 upward, with structured initialization
    for target_rank in range(18, 36):
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 7919 + target_rank * 31)
            
            # Structured initialization: start from a perturbed schoolbook
            # and project down to the target rank
            U = np.random.randn(N, target_rank) * 0.3
            V = np.random.randn(M, target_rank) * 0.3
            W = np.random.randn(P, target_rank) * 0.3
            
            # Also try starting from a subset of the schoolbook columns
            if trial % 2 == 0:
                # Pick target_rank columns from schoolbook
                cols = np.random.choice(rank_school, target_rank, replace=False)
                U = U_sch[:, cols] + np.random.randn(N, target_rank) * 0.1
                V = V_sch[:, cols] + np.random.randn(M, target_rank) * 0.1
                W = W_sch[:, cols] + np.random.randn(P, target_rank) * 0.1
            
            prev_err = float('inf')
            for it in range(60):
                if time.time() > deadline - 12:
                    break
                
                W = solve_W(U, V, target_rank)
                U = solve_U(V, W, target_rank)
                V = solve_V(U, W, target_rank)
                
                if it % 10 == 0:
                    err = error(U, V, W)
                    if err < 1e-10:
                        break
                    if abs(prev_err - err) < 1e-13:
                        break
                    prev_err = err
            
            err = error(U, V, W)
            if err < 1e-8:
                if verify(U, V, W, tol=1e-7):
                    # Try to snap to half-integers
                    def snap(arr, step=0.5, tol=1e-6):
                        s = np.round(arr / step) * step
                        if np.allclose(arr, s, atol=tol):
                            return s
                        return None
                    Us, Vs, Ws = snap(U), snap(V), snap(W)
                    if Us is not None and Vs is not None and Ws is not None:
                        if verify(Us, Vs, Ws, tol=1e-8):
                            best_result = {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                            best_rank = target_rank
                        else:
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = target_rank
                    else:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # Final check
    U_final = np.array(best_result["U"], dtype=float)
    V_final = np.array(best_result["V"], dtype=float)
    W_final = np.array(best_result["W"], dtype=float)
    
    # Ensure correct shapes
    assert U_final.shape == (N, best_rank), f"U shape {U_final.shape} != ({N}, {best_rank})"
    assert V_final.shape == (M, best_rank), f"V shape {V_final.shape} != ({M}, {best_rank})"
    assert W_final.shape == (P, best_rank), f"W shape {W_final.shape} != ({P}, {best_rank})"
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
