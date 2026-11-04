import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_np(U, V, W, rank):
        """Vectorized verification using numpy. U, V, W are numpy arrays."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def verify_np_half(U, V, W, rank):
        """Verify for half-integers. U, V, W are numpy arrays of integers (representing 2*val)."""
        # If U is 2*val, then product is 8*val. Target 8.
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 8)

    def smooth_search(target_rank, max_iters=1000, lr=0.01, seed=None):
        """
        Minimize the squared error of the tensor reconstruction.
        Returns a list of rounded solutions (integers and half-integers) as numpy arrays.
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize with small random integers
        U = np.random.randint(-1, 2, size=(nm, target_rank)).astype(float)
        V = np.random.randint(-1, 2, size=(mp, target_rank)).astype(float)
        W = np.random.randint(-1, 2, size=(pn, target_rank)).astype(float)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute error and gradients
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            
            # T_ijk = sum_c U_ijkc V_jkic W_kiic
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            error = np.sum((1.0 - T)**2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if error < 1e-6:
                    break
            
            # Gradients
            # dE/dU_ijkc = -2 * (1 - T_ijk) * V_jkc * W_kic
            # We need to compute the gradient efficiently
            # grad_U_ijkc = -2 * (1 - T_ijk) * sum? No, V and W are fixed for the gradient wrt U.
            # grad_U_ijkc = -2 * (1 - T_ijk) * V_jkc * W_kic
            # This is element-wise multiplication.
            
            # V_jkc and W_kic need to be broadcast to match U_ijkc shape (n, m, r)
            # V is (m, p, r). We need V_jkc for each i. So repeat along i axis?
            # Actually, V_jkc does not depend on i.
            # W_kic does not depend on j.
            
            # Let's compute:
            # term_V = V3 # (m, p, r)
            # term_W = W3 # (p, n, r)
            # We want grad_U3 (n, m, r)
            # grad_U3[i, j, c] = -2 * (1 - T[i, j, k?]) ... wait, T is (n, m, p).
            # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            
            # Gradient wrt U[i, j, c]:
            # dE/du = -2 * (1 - T[i, j, k]) * V[j, k, c] * W[k, i, c]
            # Sum over k? No, T[i, j, k] is for a specific k.
            # Wait, the error is sum over i, j, k of (1 - T[i, j, k])^2.
            # So dE/du[i, j, c] = sum_k -2 * (1 - T[i, j, k]) * V[j, k, c] * W[k, i, c]
            
            # This can be computed as:
            # diff = 1.0 - T # (n, m, p)
            # We need to contract diff (n, m, p) with V (m, p, r) and W (p, n, r)
            
            # grad_U[i, j, c] = sum_k diff[i, j, k] * V[j, k, c] * W[k, i, c]
            # Let's use einsum:
            # grad_U3 = -2 * np.einsum('ijk,jkc,kic->ijc', diff, V3, W3)
            
            diff = 1.0 - T
            grad_U3 = -2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V3, W3)
            
            # grad_V[j, k, c] = sum_i diff[i, j, k] * U[i, j, c] * W[k, i, c]
            grad_V3 = -2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U3, W3)
            
            # grad_W[k, i, c] = sum_j diff[i, j, k] * U[i, j, c] * V[j, k, c]
            grad_W3 = -2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U3, V3)
            
            # Gradient Descent Step
            U -= lr * grad_U3.reshape(nm, target_rank)
            V -= lr * grad_V3.reshape(mp, target_rank)
            W -= lr * grad_W3.reshape(pn, target_rank)
            
            # Normalize or clip to prevent divergence?
            # U = np.clip(U, -5, 5)
            # V = np.clip(V, -5, 5)
            # W = np.clip(W, -5, 5)
        
        # Round to nearest integer and half-integer
        candidates = []
        
        # Round to nearest integer
        U_int = np.round(best_U).astype(np.int64)
        V_int = np.round(best_V).astype(np.int64)
        W_int = np.round(best_W).astype(np.int64)
        candidates.append((U_int, V_int, W_int, "int"))
        
        # Round to nearest half-integer (store as 2*val)
        U_half = np.round(best_U * 2).astype(np.int64)
        V_half = np.round(best_V * 2).astype(np.int64)
        W_half = np.round(best_W * 2).astype(np.int64)
        candidates.append((U_half, V_half, W_half, "half"))
        
        return candidates

    def local_search_int(U, V, W, target_rank, max_iters=200):
        """
        Local search on integer entries to fix small errors.
        U, V, W are numpy arrays of int64.
        """
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if verify_np(U, V, W, target_rank):
                return U, V, W
            
            # Pick a random matrix and element
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm - 1)
                col = random.randint(0, target_rank - 1)
                old = U[row, col]
                U[row, col] += random.choice([-1, 1])
                # Check if error decreased
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.sum((1.0 - T)**2)
                # We don't have previous error. Let's just revert if not valid.
                # This is a heuristic.
                if not verify_np(U, V, W, target_rank):
                    U[row, col] = old
            elif mat == 'V':
                row = random.randint(0, mp - 1)
                col = random.randint(0, target_rank - 1)
                old = V[row, col]
                V[row, col] += random.choice([-1, 1])
                if not verify_np(U, V, W, target_rank):
                    V[row, col] = old
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, target_rank - 1)
                old = W[row, col]
                W[row, col] += random.choice([-1, 1])
                if not verify_np(U, V, W, target_rank):
                    W[row, col] = old
        return None

    # Try ranks from 21 to 41
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            
            candidates = smooth_search(target_rank, max_iters=500, lr=0.05, seed=restart)
            
            for U_cand, V_cand, W_cand, typ in candidates:
                if typ == "int":
                    if verify_np(U_cand, V_cand, W_cand, target_rank):
                        best = {"U": U_cand.tolist(), "V": V_cand.tolist(), "W": W_cand.tolist()}
                        best_rank = target_rank
                        found = True
                        break
                    # Try local search
                    res = local_search_int(U_cand.copy(), V_cand.copy(), W_cand.copy(), target_rank, max_iters=100)
                    if res:
                        best = {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}
                        best_rank = target_rank
                        found = True
                        break
                else: # half
                    if verify_np_half(U_cand, V_cand, W_cand, target_rank):
                        # Convert to list of floats (half-integers)
                        best = {"U": (U_cand / 2.0).tolist(), "V": (V_cand / 2.0).tolist(), "W": (W_cand / 2.0).tolist()}
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break

    return best
# EVOLVE-BLOCK-END
