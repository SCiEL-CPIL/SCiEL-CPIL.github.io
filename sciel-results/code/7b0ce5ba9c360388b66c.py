"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    T_target = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u = np.array(U[:, c], dtype=float).reshape(n, m)
        v = np.array(V[:, c], dtype=float).reshape(m, p)
        w = np.array(W[:, c], dtype=float).reshape(p, n)
        T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _check_half_int(U, V, W):
    """Check all entries are integers or half-integers."""
    for mat in (U, V, W):
        for row in mat:
            for x in row:
                v = float(x)
                if abs(v * 2 - round(v * 2)) > 1e-6:
                    return False
    return True

def _snap_to_half_int(mat):
    """Snap values to nearest half-integer."""
    return [[round(x * 2) / 2.0 for x in row] for row in mat]

def _build_tensor(n, m, p):
    """Build the all-ones tensor."""
    return np.ones((n, m, p))

def _compute_tensor(U, V, W, n, m, p, rank):
    """Compute the reconstructed tensor."""
    T = np.zeros((n, m, p))
    for c in range(rank):
        u = np.array(U[:, c], dtype=float).reshape(n, m)
        v = np.array(V[:, c], dtype=float).reshape(m, p)
        w = np.array(W[:, c], dtype=float).reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return T

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Schoolbook baseline: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    T_target = _build_tensor(n, m, p)

    # Strategy: For each target rank, try random linear combinations of schoolbook terms.
    # Each term in a decomposition is a linear combination of the 24 schoolbook terms.
    # We pick random coefficients, snap to half-integers, and verify.
    # 
    # A decomposition of rank r can be written as:
    #   sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where each u_c, v_c, w_c are random linear combinations of the basis vectors.
    # 
    # Alternative: use the fact that the tensor is all-ones. Try structured decompositions.

    # First, try a very direct algebraic approach:
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try rank from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # = 12

    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break

        found = False
        max_trials = 20000
        trial = 0

        while trial < max_trials and not found:
            if time.time() > deadline - 5:
                break

            rng = np.random.RandomState(trial)

            # Generate random rank-r decomposition via random matrices
            # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
            # Use random integer/half-integer entries in a small range
            # This is key: restrict to half-integers directly
            
            scale = 2  # entries in {-1, -0.5, 0, 0.5, 1} or similar
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))
            W_rand = np.zeros((p * n, target_rank))

            for _ in range(target_rank):
                # Random half-integer entries
                U_rand[:, _] = rng.randint(-2, 3, size=n * m) / 2.0
                V_rand[:, _] = rng.randint(-2, 3, size=m * p) / 2.0
                W_rand[:, _] = rng.randint(-2, 3, size=p * n) / 2.0

            T_recon = _compute_tensor(U_rand, V_rand, W_rand, n, m, p, target_rank)
            if np.max(np.abs(T_recon - T_target)) < 1e-9:
                # Verify with exact check
                U_list = [[U_rand[i, c] for c in range(target_rank)] for i in range(n * m)]
                V_list = [[V_rand[i, c] for c in range(target_rank)] for i in range(m * p)]
                W_list = [[W_rand[i, c] for c in range(target_rank)] for i in range(p * n)]
                if _check_half_int(U_list, V_list, W_list) and _validate(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break

            trial += 1

        if found:
            break

    # If random search didn't find anything, try a structured approach:
    # Use the fact that T is all-ones. Try decompositions where
    # U[i*m+j, c] depends only on i, V[j*p+k, c] depends only on j, etc.
    # This gives rank at most n + m + p - 2 or similar.
    # Actually, a rank-1 term u⊗v⊗w where u depends on i, v on j, w on k
    # gives T[i,j,k] = sum_c u_c(i) * v_c(j) * w_c(k)
    # We need this to equal 1 for all i,j,k.
    # This is possible with rank 1: u(i)=1, v(j)=1, w(k)=1 gives T=1.
    # But wait - the indices are coupled! U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The index j appears in both U and V, k in V and W, i in W and U.
    # So we can't separate them that simply.

    # Let's try a different structured approach:
    # Group by the "cycle" i->j->k->i
    # For each (i,j,k), the term is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # 
    # Try: for each c, let U[i*m+j,c] = a_c(i,j), V[j*p+k,c] = b_c(j,k), W[k*n+i,c] = c_c(k,i)
    # We need sum_c a_c(i,j) * b_c(j,k) * c_c(k,i) = 1 for all i,j,k

    # A simple construction: use the "cycle" structure
    # For n=2, m=3, p=4:
    # i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    
    # Try rank = n*m*p / 2 = 12 with clever pairing
    # Or try the known result: for m*p matrix mult, rank can sometimes be reduced.

    # Let me try a more systematic search with smaller entry ranges
    if best_rank == rank_sb:
        for target_rank in range(lower_bound, min(best_rank, 16)):
            if time.time() > deadline - 10:
                break

            found = False
            # Use smaller entry range for more structured search
            max_trials = 50000
            trial = 0

            while trial < max_trials and not found:
                if time.time() > deadline - 5:
                    break

                rng = np.random.RandomState(100000 + trial)

                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))
                W_rand = np.zeros((p * n, target_rank))

                # Use entries from {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
                for _ in range(target_rank):
                    U_rand[:, _] = rng.randint(-1, 2, size=n * m) / 2.0
                    V_rand[:, _] = rng.randint(-1, 2, size=m * p) / 2.0
                    W_rand[:, _] = rng.randint(-1, 2, size=p * n) / 2.0

                T_recon = _compute_tensor(U_rand, V_rand, W_rand, n, m, p, target_rank)
                if np.max(np.abs(T_recon - T_target)) < 1e-9:
                    U_list = [[U_rand[i, c] for c in range(target_rank)] for i in range(n * m)]
                    V_list = [[V_rand[i, c] for c in range(target_rank)] for i in range(m * p)]
                    W_list = [[W_rand[i, c] for c in range(target_rank)] for i in range(p * n)]
                    if _check_half_int(U_list, V_list, W_list) and _validate(U_list, V_list, W_list, n, m, p):
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break

                trial += 1

            if found:
                break

    return best_result
# EVOLVE-BLOCK-END
