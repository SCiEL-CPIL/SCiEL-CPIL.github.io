"""
Matrix multiplication 3x4x5 tensor rank minimization.
Schoolbook gives rank 60. We try to find a lower rank exact decomposition.
"""
import numpy as np
import time
from fractions import Fraction
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    deadline = time.time() + 130  # safety margin before 150s kill

    # Schoolbook as baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def _to_frac(x):
        """Convert float to Fraction if it's a nice half-integer or integer."""
        f = Fraction(x).limit_denominator(1000)
        # Check if denominator is 1 or 2
        if f.denominator in (1, 2):
            return f
        return None

    def _verify(U, V, W, n, m, p, tol=1e-10):
        """Check if T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] equals delta_{i,k}."""
        # Actually the tensor for matrix mult: C_{i,k} = sum_j A_{i,j} B_{j,k}
        # So the tensor T_{i,j,k} = 1 (it's always 1 for valid i,j,k in range)
        # Wait no. The multiplication tensor has entries T_{i,j,k} where the
        # product C_{i,k} gets contribution from A_{i,j} * B_{j,k}.
        # So T_{i,j,k} = 1 for all i,j,k (each triple contributes to exactly one output).
        # The decomposition: T_{i,j,k} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
        # We need this to equal 1 for all i in [0,n), j in [0,m), k in [0,p).
        rank = U.shape[1]
        # Build tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _verify_fast(U, V, W, n, m, p):
        """Faster verification using numpy. T[i,j,k] should be 1 for all i,j,k."""
        rank = U.shape[1]
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's compute all at once using einsum
        # U[i*m+j, r] -> index (i,j,r)
        # V[j*p+k, r] -> index (j,k,r)
        # W[k*n+i, r] -> index (k,i,r)
        U3 = U.reshape(n, m, rank)  # U[i,j,r]
        V3 = V.reshape(m, p, rank)  # V[j,k,r]
        W3 = W.reshape(p, n, rank)  # W[k,i,r]
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3, optimize=True)
        # T should be all 1s
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
        return True

    def _to_frac_list(mat):
        """Convert a matrix to list of lists, trying to find rational entries."""
        result = []
        for row in mat:
            frow = []
            for x in row:
                f = _to_frac(x)
                if f is not None:
                    frow.append(f)
                else:
                    frow.append(Fraction(x).limit_denominator(10000))
            result.append(frow)
        return result

    # Try random perturbation + rounding approach
    # Start from schoolbook and try to merge terms
    # Strategy: use random rank-reduction via low-rank approximation

    random.seed(42)
    np.random.seed(42)

    # Try various ranks from 59 down
    for target_rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
        if time.time() > deadline - 10:
            break

        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break

            # Initialize U, V, W randomly with small values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1

            # Iteratively adjust to reduce error
            # Error: T[i,j,k] - 1
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break

                # Compute current tensor
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
                error = T - 1.0

                # Update U: gradient w.r.t. U[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] * error[i,j,k]
                # Sum over k
                # dU[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r] * error[i,j,k]
                dU = np.zeros((nm, target_rank))
                for i in range(n):
                    for j in range(m):
                        for r in range(target_rank):
                            s = 0.0
                            for k in range(p):
                                s += V[j*p+k, r] * W[k*n+i, r] * error[i, j, k]
                            dU[i*m+j, r] = s

                dV = np.zeros((mp, target_rank))
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            s = 0.0
                            for i in range(n):
                                s += U[i*m+j, r] * W[k*n+i, r] * error[i, j, k]
                            dV[j*p+k, r] = s

                dW = np.zeros((pn, target_rank))
                for k in range(p):
                    for i in range(n):
                        for r in range(target_rank):
                            s = 0.0
                            for j in range(m):
                                s += U[i*m+j, r] * V[j*p+k, r] * error[i, j, k]
                            dW[k*n+i, r] = s

                # Gradient descent
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

            # After gradient descent, check if we got close enough
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
            max_err = np.max(np.abs(T - 1.0))

            if max_err < 1e-6:
                # Try to round to nice fractions
                U_f = np.round(U * 2) / 2  # half-integers
                V_f = np.round(V * 2) / 2
                W_f = np.round(W * 2) / 2

                U3f = U_f.reshape(n, m, target_rank)
                V3f = V_f.reshape(m, p, target_rank)
                W3f = W_f.reshape(p, n, target_rank)
                Tf = np.einsum('ijr,jkr,kir->ijk', U3f, V3f, W3f, optimize=True)
                if np.allclose(Tf, 1.0, atol=1e-10):
                    best_rank = target_rank
                    best_U = U_f
                    best_V = V_f
                    best_W = W_f
                    break

            if time.time() > deadline - 10:
                break

    # Try integer-only rounding
    for target_rank in range(best_rank - 1, max(best_rank - 5, 45), -1):
        if time.time() > deadline - 5:
            break

        for attempt in range(30):
            if time.time() > deadline - 5:
                break

            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)

            for iteration in range(300):
                if time.time() > deadline - 5:
                    break
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
                error = T - 1.0

                dU = np.zeros((nm, target_rank))
                for i in range(n):
                    for j in range(m):
                        for r in range(target_rank):
                            dU[i*m+j, r] = sum(
                                V[j*p+k, r] * W[k*n+i, r] * error[i, j, k] for k in range(p)
                            )
                dV = np.zeros((mp, target_rank))
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            dV[j*p+k, r] = sum(
                                U[i*m+j, r] * W[k*n+i, r] * error[i, j, k] for i in range(n)
                            )
                dW = np.zeros((pn, target_rank))
                for k in range(p):
                    for i in range(n):
                        for r in range(target_rank):
                            dW[k*n+i, r] = sum(
                                U[i*m+j, r] * V[j*p+k, r] * error[i, j, k] for j in range(m)
                            )

                lr = 0.005
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

            # Round to integers
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)

            U3r = U_r.reshape(n, m, target_rank)
            V3r = V_r.reshape(m, p, target_rank)
            W3r = W_r.reshape(p, n, target_rank)
            Tr = np.einsum('ijr,jkr,kir->ijk', U3r, V3r, W3r, optimize=True)
            if np.allclose(Tr, 1.0, atol=1e-10):
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break

    # Final verification and return
    if best_rank < rank_school:
        U_final = best_U
        V_final = best_V
        W_final = best_W
    else:
        U_final = U_school
        V_final = V_school
        W_final = W_school

    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
