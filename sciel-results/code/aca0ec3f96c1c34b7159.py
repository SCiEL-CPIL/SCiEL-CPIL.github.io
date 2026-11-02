"""Matrix multiplication 2x4x5 rank-minimal tensor decomposition.
Block-Strassen base + systematic column removal + random refinement."""

import numpy as np
import itertools
import random
import time

DIMENSIONS = [2, 4, 5]
START_TIME = time.time()
DEADLINE = 150.0
SAFETY = 5.0  # seconds margin

def _time_left():
    return DEADLINE - SAFETY - (time.time() - START_TIME)

def construct():
    n, m, p = 2, 4, 5
    N, M, P = n * m, m * p, p * n  # 8, 20, 10
    np.random.seed(0)
    random.seed(0)

    # --- Build the multiplication tensor T[i,j,k] = 1 iff output uses A[i,j]*B[j,k] ---
    # Output C[i,k] = sum_j A[i,j] B[j,k]. Tensor index: T is a 3-tensor.
    # We represent the matrix multiplication as: for each (i,j,k) with valid indices,
    # U[i,j] * V[j,k] * W[i,k] contributes.
    # U shape (N, rank) indexed by (i,j), V shape (M, rank) by (j,k), W shape (P, rank) by (k,i)
    # Trace convention: W[k*n+i, r]

    # Build tensor T as a 3D array: T[(i,j),(j,k),(k,i)] = 1
    # Actually the matrix multiplication tensor: T[i, j, k] where the three modes are
    # input1 (i,j), input2 (j,k), output (i,k).
    # T is nonzero at ( (i,j), (j,k), (i,k) ) for each valid (i,j,k).
    # i in [0,n), j in [0,m), k in [0,p)
    # Mode 1: (i,j) -> i*m + j, size N=8
    # Mode 2: (j,k) -> j*p + k, size M=20
    # Mode 3: (i,k) -> k*n + i (trace convention W[k*n+i]), size P=10

    # Build the tensor as a list of (u_idx, v_idx, w_idx) triples
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_indices.append((u, v, w))

    # --- Block Strassen construction ---
    # For 2x4x5: split A (2x4) into two 2x2 blocks, B (4x5) into two 2x5 blocks.
    # C = A1*B1 + A2*B2 where A1=A[0:2,:], A2=A[2:4,:], B1=B[:2,:], B2=B[2:,:].
    # Wait, A is 2x4, B is 4x5. So A has 2 rows, 4 cols. B has 4 rows, 5 cols.
    # Split the 4 middle dimension: A = [A_left | A_right] (2x2 each), B = [B_top; B_bottom] (2x5 each)
    # C = A_left * B_top + A_right * B_bottom
    # Each product is 2x2 * 2x5 = 2x5.
    # Use Strassen for 2x2 * 2x2? No, B is 2x5. Let's use a different approach.
    # Actually let's just use the schoolbook for the base and then try to reduce.
    # Schoolbook rank = n*m*p = 2*4*5 = 40.

    # Let me try a smarter base: split middle into 2+2.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each term is 2x2 @ 2x5. Rank of each = 2*2*5 = 20. Total = 40.
    # That's the same as schoolbook.

    # Alternative: use rank-3 for 2x2x2 blocks.
    # Split into 2x2x5: treat B as two 2x2 blocks + one 2x1? Messy.
    # Let's just start with schoolbook rank 40 and try to reduce.

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        for c, (u, v, w) in enumerate(tensor_indices):
            U[u, c] = 1
            V[v, c] = 1
            W[w, c] = 1
        return U, V, W

    U, V, W = make_schoolbook()
    best_rank = U.shape[1]

    # --- Verification: check that sum_r U[:,r]*V[:,r]*W[:,r] reconstructs tensor ---
    def verify(U, V, W, tol=1e-9):
        # Reconstruct tensor: for each (u,v,w), check sum_r U[u,r]*V[v,r]*W[w,r] == T[u,v,w]
        # Build sparse tensor
        T = {}
        for (u, v, w) in tensor_indices:
            T[(u, v, w)] = 1
        # Check nonzero entries
        for (u, v, w) in tensor_indices:
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            if abs(val - 1) > tol:
                return False
        # Check zero entries (sample)
        all_uvw = set()
        for u in range(N):
            for v in range(M):
                for w in range(P):
                    all_uvw.add((u, v, w))
        nonzero = set(tensor_indices)
        zero_entries = all_uvw - nonzero
        # Sample zero entries
        zero_list = list(zero_entries)
        random.shuffle(zero_list)
        for (u, v, w) in zero_list[:200]:
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            if abs(val) > tol:
                return False
        return True

    # --- Try to reduce rank by removing columns ---
    def try_remove_columns(U, V, W):
        """Try removing columns one at a time, checking validity."""
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        improved = True
        while improved and _time_left() > 1:
            improved = False
            for c in range(U.shape[1]):
                if _time_left() < 1:
                    break
                # Try removing column c
                U2 = np.delete(U, c, axis=1)
                V2 = np.delete(V, c, axis=1)
                W2 = np.delete(W, c, axis=1)
                if verify(U2, V2, W2):
                    U, V, W = U2, V2, W2
                    improved = True
                    break
        return U, V, W

    # --- Random perturbation + re-optimization ---
    def random_refine(U, V, W, n_trials=500):
        """Try random perturbations of columns to enable more removals."""
        U, V, W = U.copy(), V.copy(), W.copy()
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]

        for trial in range(n_trials):
            if _time_left() < 2:
                break
            # Pick a random subset of columns to perturb
            rank = U.shape[1]
            n_perturb = random.randint(1, min(3, rank))
            cols = random.sample(range(rank), n_perturb)
            U_p = U.copy()
            V_p = V.copy()
            W_p = W.copy()
            for c in cols:
                # Add small random perturbation
                U_p[:, c] += np.random.randn(N) * 0.01
                V_p[:, c] += np.random.randn(M) * 0.01
                W_p[:, c] += np.random.randn(P) * 0.01
            # Try to remove columns
            U_r, V_r, W_r = try_remove_columns(U_p, V_p, W_p)
            if U_r.shape[1] < best_rank:
                best_rank = U_r.shape[1]
                best_U, best_V, best_W = U_r, V_r, W_r
        return best_U, best_V, best_W

    # --- Greedy: try to find linear dependencies among columns ---
    def find_linear_dependence(U, V, W):
        """Check if any column is a linear combination of others (in the tensor sense)."""
        rank = U.shape[1]
        # A column c is redundant if the tensor contribution can be absorbed.
        # This is hard to check directly. Instead, try removing each column.
        for c in range(rank):
            if _time_left() < 1:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if verify(U2, V2, W2):
                return c
        return -1

    # --- Main search ---
    # Start with schoolbook
    U, V, W = make_schoolbook()
    best_rank = U.shape[1]

    # Phase 1: Systematic column removal
    U, V, W = try_remove_columns(U, V, W)
    best_rank = U.shape[1]

    # Phase 2: Random refinement
    U, V, W = random_refine(U, V, W, n_trials=300)
    best_rank = U.shape[1]

    # Phase 3: More systematic search with different random seeds
    for seed in range(1, 20):
        if _time_left() < 3:
            break
        np.random.seed(seed)
        random.seed(seed)
        U_s, V_s, W_s = make_schoolbook()
        # Perturb randomly
        rank = U_s.shape[1]
        for c in range(rank):
            U_s[:, c] += np.random.randn(N) * 0.005
            V_s[:, c] += np.random.randn(M) * 0.005
            W_s[:, c] += np.random.randn(P) * 0.005
        U_r, V_r, W_r = try_remove_columns(U_s, V_s, W_s)
        if U_r.shape[1] < best_rank:
            best_rank = U_r.shape[1]
            U, V, W = U_r, V_r, W_r

    # Final verification
    if not verify(U, V, W):
        # Fallback to schoolbook
        U, V, W = make_schoolbook()
        best_rank = U.shape[1]

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
