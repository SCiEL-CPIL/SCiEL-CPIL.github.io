"""Matrix multiplication 3,3,6 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
N, M, P = 3, 3, 6
DEADLINE = 120.0  # wall-clock limit with safety margin


def _build_tensor():
    """Build the matmul tensor T[i,j,k] for A(i,j) * B(j,k) -> C(i,k).
    We need to find rank-r decomposition:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    T = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-6):
    """Check if U,V,W decompose T. U is (n*m, r), V is (m*p, r), W is (p*n, r)."""
    n, m, p = N, M, P
    r = U.shape[1]
    # Reconstruct tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - T[i, j, k]) > tol:
                    return False
    return True


def _verify_fast(U, V, W, T, tol=1e-6):
    """Vectorized verification."""
    n, m, p = N, M, P
    r = U.shape[1]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, T, atol=tol)


def _als_step(T, U, V, W, tol=1e-8, max_iter=500):
    """ALS (Alternating Least Squares) for tensor decomposition.
    Minimize ||T - sum_c U[:,c]*V[:,c]*W[:,c]||^2 by alternating.
    """
    n, m, p = N, M, P
    r = U.shape[1]
    
    # Precompute for efficiency
    # T is (n, m, p)
    
    for it in range(max_iter):
        # Update U: for each (i,j), U[i*m+j, :] minimizes
        # sum_k (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
        # This is a linear system: for fixed (i,j),
        # T[i,j,:] = U[i*m+j,:] contracted with V[j,:,:] and W[:,i,:]
        # More precisely: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Let a_c = U[i*m+j,c], b_c(k) = V[j*p+k,c]*W[k*n+i,c]
        # Then T[i,j,k] = sum_c a_c * b_c(k)
        # So [b_0(k), b_1(k), ..., b_{r-1}(k)] * a = T[i,j,:]
        # B_mat (p x r): B_mat[k,c] = V[j*p+k,c] * W[k*n+i,c]
        
        for i in range(n):
            for j in range(m):
                # Build p x r matrix
                B_mat = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        B_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                # Solve B_mat @ a = T[i,j,:]
                try:
                    a, _, _, _ = np.linalg.lstsq(B_mat, T[i, j, :], rcond=None)
                except:
                    continue
                U[i * m + j, :] = a
        
        # Update V: for each (j,k), V[j*p+k, :] minimizes
        # T[i,j,k] over i. T[:,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For fixed (j,k): T[i,j,k] = sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c]
        # Let a_c = V[j*p+k,c], b_c(i) = U[i*m+j,c]*W[k*n+i,c]
        # C_mat (n x r): C_mat[i,c] = U[i*m+j,c]*W[k*n+i,c]
        # C_mat @ a = T[:,j,k]
        
        for j in range(m):
            for k in range(p):
                C_mat = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        C_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                try:
                    a, _, _, _ = np.linalg.lstsq(C_mat, T[:, j, k], rcond=None)
                except:
                    continue
                V[j * p + k, :] = a
        
        # Update W: for each (k,i), W[k*n+i, :] minimizes
        # T[i,j,k] over j. T[i,:,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For fixed (k,i): T[i,j,k] = sum_c W[k*n+i,c] * U[i*m+j,c]*V[j*p+k,c]
        # Let a_c = W[k*n+i,c], b_c(j) = U[i*m+j,c]*V[j*p+k,c]
        # D_mat (m x r): D_mat[j,c] = U[i*m+j,c]*V[j*p+k,c]
        # D_mat @ a = T[i,:,k]
        
        for k in range(p):
            for i in range(n):
                D_mat = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        D_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                try:
                    a, _, _, _ = np.linalg.lstsq(D_mat, T[i, :, k], rcond=None)
                except:
                    continue
                W[k * n + i, :] = a
        
        # Check convergence
        if _verify_fast(U, V, W, T, tol=1e-6):
            return True
    
    return _verify_fast(U, V, W, T, tol=1e-6)


def _try_rank(r, T, max_restarts=20, time_limit=60.0):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random values
        rng = np.random.RandomState(restart * 1000 + r)
        U = rng.randn(N * M, r) * 0.1
        V = rng.randn(M * P, r) * 0.1
        W = rng.randn(P * N, r) * 0.1
        
        # Run ALS
        if _als_step(T, U, V, W, tol=1e-8, max_iter=200):
            # Verify with higher precision
            if _verify_fast(U, V, W, T, tol=1e-10):
                return U, V, W
    
    return None


def _try_rank_with_perturbation(r, T, max_restarts=10, time_limit=30.0):
    """Try rank-r with structured initializations."""
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(restart * 7919 + r * 13)
        
        # Different initialization strategies
        if restart % 3 == 0:
            U = rng.randn(N * M, r) * 0.5
            V = rng.randn(M * P, r) * 0.5
            W = rng.randn(P * N, r) * 0.5
        elif restart % 3 == 1:
            # Sparse initialization
            U = rng.choice([-1, 0, 1], size=(N * M, r))
            V = rng.choice([-1, 0, 1], size=(M * P, r))
            W = rng.choice([-1, 0, 1], size=(P * N, r))
        else:
            U = rng.randn(N * M, r)
            V = rng.randn(M * P, r)
            W = rng.randn(P * N, r)
        
        if _als_step(T, U, V, W, tol=1e-8, max_iter=300):
            if _verify_fast(U, V, W, T, tol=1e-10):
                return U, V, W
    
    return None


def _schoolbook():
    """Return the schoolbook decomposition as fallback."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    T = _build_tensor()
    
    # The tensor for 3x3 * 3x6 matmul has all entries = 1.
    # This is a very structured tensor. The schoolbook rank is 54.
    # We try to find lower rank decompositions.
    
    # For the all-ones tensor T[i,j,k]=1, we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # A simple low-rank construction: if we can find vectors such that
    # the product sums to 1 uniformly.
    
    # Try rank 1: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1 for all i,j,k
    # This requires U[i*m+j,0] = a(i,j), V[j*p+k,0]=b(j,k), W[k*n+i,0]=c(k,i)
    # a(i,j)*b(j,k)*c(k,i) = 1 for all i,j,k
    # Let a(i,j) = alpha_i * beta_j, b(j,k) = gamma_j * delta_k, c(k,i) = epsilon_k * zeta_i
    # Then product = alpha_i * beta_j * gamma_j * delta_k * epsilon_k * zeta_i
    # = (alpha_i * zeta_i) * (beta_j * gamma_j) * (delta_k * epsilon_k)
    # For this to be 1 for all i,j,k, we need alpha_i*zeta_i = 1, beta_j*gamma_j=1, delta_k*epsilon_k=1
    # So rank 1 works! U[i*m+j,0] = alpha_i*beta_j, V[j*p+k,0]=gamma_j*delta_k, W[k*n+i,0]=epsilon_k*zeta_i
    # with alpha_i*zeta_i=1, beta_j*gamma_j=1, delta_k*epsilon_k=1
    
    # Let's verify: set alpha_i=1, zeta_i=1, beta_j=1, gamma_j=1, delta_k=1, epsilon_k=1
    # Then U=1, V=1, W=1, product = 1. Yes!
    
    # So rank 1 should work for the all-ones tensor!
    U = [[1] for _ in range(N * M)]
    V = [[1] for _ in range(M * P)]
    W = [[1] for _ in range(P * N)]
    
    # Verify
    if _verify_fast(np.array(U), np.array(V), np.array(W), T, tol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # If that didn't work (shouldn't happen), fall back to schoolbook
    return _schoolbook()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
