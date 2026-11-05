"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    def to_half_integer(v, tol=1e-9):
        """Round to nearest half-integer if close."""
        rounded = np.round(v * 2) / 2
        if np.max(np.abs(v - rounded)) < tol:
            return rounded
        return v

    def validate(U, V, W):
        """Check that U*V*W reconstructs the tensor T[i,j,k] = 1."""
        n, m, p = DIMENSIONS
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Check a subset of (i,j,k)
        checks = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum over c
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.dot(u_val, v_val)
                    s = s @ w_val  # Actually need element-wise: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                    # Redo: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-7:
                        return False
                    checks += 1
        return True

    def schoolbook():
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
        return U, V, W

    # Start with schoolbook
    U, V, W = schoolbook()
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find lower rank via random initialization + ALS-style refinement
    # The tensor T[i,j,k] = delta(i,j,k all valid) = 1 for all i,j,k
    # This is the "all-ones" tensor, which has rank related to matrix multiplication.
    # For matrix multiplication, the tensor is T[i,j,k] = 1 (it's the identity of multiplication).
    # Actually for (n,m,p) matrix mult, the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1. This is the 3-way all-ones tensor.

    # The all-ones tensor of size n x m x p has rank 1! (U[i,j]=1, V[j,k]=1, W[k,i]=1)
    # Wait, that's not right for matrix multiplication. Let me reconsider.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor for this is T[i,j,k] = 1, meaning the bilinear form is sum_{i,j,k} C[i,k] * T[i,j,k] * A[i,j] * B[j,k]
    # Actually the multiplication tensor is T where T[i,j,k] = 1 for all i,j,k, and the
    # decomposition gives C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[r] ... no.
    # Standard: C[i,k] = sum_j A[i,j] B[j,k] = sum_r sum_j (U[i,j,r] A[i,j]) (V[j,k,r] B[j,k]) W[r,k,i]
    # So T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] should equal 1 for all i,j,k.
    # The all-ones tensor indeed has this property. Rank of all-ones tensor [n,m,p] is 1?
    # No! rank 1 means T[i,j,k] = a_i * b_j * c_k. All-ones = 1*1*1, so yes rank 1.
    # But that can't be right for matrix multiplication...
    # Actually for matrix multiplication the tensor IS the all-ones tensor, and its rank
    # equals the minimum number of bilinear multiplications needed. The all-ones tensor
    # T[i,j,k]=1 has rank 1 in the tensor sense? No, tensor rank (CP rank) of all-ones
    # [n,m,p] is 1 only if we allow arbitrary factors. 1*1*1=1, so CP rank is 1.
    # But matrix multiplication rank is NOT 1. The issue is that the tensor for matrix
    # multiplication is NOT the all-ones tensor in the standard CP sense.
    # Let me re-read: T[i*m+j, j*p+k, k*n+i] = 1. The factors are indexed by (i,j), (j,k), (k,i).
    # So it's T[a,b,c] where a=(i,j), b=(j,k), c=(k,i), and T=1 when these indices are
    # consistent (i from a matches i from c, j from a matches j from b, k from b matches k from c).
    # This is NOT the all-ones tensor on [nm, mp, pn]. It's a structured tensor.

    # For (3,5,5): indices a in [0,15), b in [0,25), c in [0,15)
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for consistent i,j,k
    # This is the matrix multiplication tensor.

    # Known: rank of (2,2,2) matmul is 7 (Strassen).
    # For (3,5,5), we need to search.

    # Let's try randomized search with ALS refinement
    random.seed(42)
    np.random.seed(42)

    for attempt in range(50):
        if time.time() > deadline - 20:
            break

        # Try various ranks
        for rank in [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]:
            if rank >= best_rank:
                continue
            if time.time() > deadline - 20:
                break

            # Random initialization
            U_r = np.random.randn(n * m, rank)
            V_r = np.random.randn(m * p, rank)
            W_r = np.random.randn(p * n, rank)

            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                # Update U
                for a in range(n * m):
                    i, j = divmod(a, m)
                    # T[a, :, :] involves b=(j,k) for all k, c=(k,i) for all k
                    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                    # For fixed a, U[a,:] is the variable.
                    # min ||sum_r U[a,r] (V[:,r] * W[:,r]) - T[a,:,:]||
                    # For each b,c: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                    # This is a linear system in U[a,:]
                    # Let's build the system: for all (b,c) where T[a,b,c] is defined
                    # T[a,b,c]=1 when b=(j,k) and c=(k,i) for consistent k
                    # b = j*p+k, c = k*n+i for k in range(p)
                    rhs = np.zeros(rank)
                    # For each k: b = j*p+k, c = k*n+i
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        # T[a,b,c] = 1
                        # sum_r U[a,r] * V[b,r] * W[c,r] = 1
                        rhs += V[b, :] * W[c, :]
                    # Solve: V*W @ U[a,:] = rhs where (V*W)[b,c,r] = V[b,r]*W[c,r]
                    # Actually we need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                    # Stack all (b,c) pairs: for k in range(p): b=j*p+k, c=k*n+i
                    # This gives p equations in rank unknowns
                    # If rank > p, underdetermined; use least squares
                    # Build matrix: rows are (b,c) pairs, cols are r
                    # M[r] = sum over (b,c) of V[b,r]*W[c,r]*delta(T[a,b,c]=1)
                    # Actually: for each (b,c) with T=1: sum_r U[a,r] V[b,r] W[c,r] = 1
                    # This is: (V[b,:] * W[c,:]) @ U[a,:] = 1 for each (b,c)
                    # Stack: A @ U[a,:] = ones, where A has rows V[b,r]*W[c,r]
                    A = np.zeros((p, rank))
                    for idx, k in enumerate(range(p)):
                        b = j * p + k
                        c = k * n + i
                        A[idx, :] = V_r[b, :] * W_r[c, :]
                    # Solve A @ U[a,:] = ones(p)
                    try:
                        U_r[a, :], residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except:
                        pass

                # Update V
                for b in range(m * p):
                    j, k = divmod(b, p)
                    # T[a,b,c]=1 when a=(i,j), c=(k,i) for i in range(n)
                    A = np.zeros((n, rank))
                    for idx, i in enumerate(range(n)):
                        a = i * m + j
                        c = k * n + i
                        A[idx, :] = U_r[a, :] * W_r[c, :]
                    try:
                        V_r[b, :], residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except:
                        pass

                # Update W
                for c in range(p * n):
                    k, i = divmod(c, n)
                    # T[a,b,c]=1 when a=(i,j), b=(j,k) for j in range(m)
                    A = np.zeros((m, rank))
                    for idx, j in enumerate(range(m)):
                        a = i * m + j
                        b = j * p + k
                        A[idx, :] = U_r[a, :] * V_r[b, :]
                    try:
                        W_r[c, :], residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        pass

                # Check convergence
                max_diff = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U_r[a, :] * V_r[b, :] * W_r[c, :])
                            max_diff = max(max_diff, abs(val - 1.0))
                if max_diff < 1e-6:
                    converged = True
                    break

            if converged or max_diff < 1e-4:
                # Try to snap to half-integers
                U_snap = to_half_integer(U_r)
                V_snap = to_half_integer(V_r)
                W_snap = to_half_integer(W_r)

                # Validate
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U_snap[a, :] * V_snap[b, :] * W_snap[c, :])
                            if abs(val - 1.0) > 1e-7:
                                valid = False
                                break
                    if not valid:
                        break

                if valid and rank < best_rank:
                    best_rank = rank
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    # Try even lower ranks
                    break
        else:
            continue
        break

    return best_result
# EVOLVE-BLOCK-END
