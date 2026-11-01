"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that the factors exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For correct decomposition, T[i,j,k] should be 1 for all (i,j,k)
    # Actually, the tensor for matrix multiplication C = A@B is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear form is: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] is the output factor.
    # The decomposition is: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'} for all i,j,j',k
    # Actually more precisely, the tensor T[i,j,k] = sum_j A[i,j] B[j,k] can be written as
    # sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j V[j*p+k,r] B[j,k]) W[k*n+i,r]
    # Expanding: sum_r sum_j sum_j' U[i*m+j,r] V[j'*p+k,r] W[k*n+i,r] A[i,j] B[j',k]
    # This must equal sum_j A[i,j] B[j,k] = sum_j sum_j' delta_{j,j'} A[i,j] B[j',k]
    # So we need: sum_r U[i*m+j,r] V[j'*p+k,r] W[k*n+i,r] = delta_{j,j'} for all i,j,j',k
    
    # Let's verify with random matrices
    rng = np.random.default_rng(42)
    for _ in range(5):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    acc += W[kk * n + ii, cc] * sum(U[ii * m + jj, cc] * A[ii, jj] for jj in range(m)) * sum(V[jj * p + kk, cc] * B[jj, kk] for jj in range(m))
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=tol):
            return False
    return True

def _best_result(n, m, p, rank, elapsed_limit=110.0):
    """Return the best known decomposition of the given rank, or None if not found."""
    t0 = time.time()
    while time.time() - t0 < elapsed_limit:
        # Try to find a rank-`rank` decomposition
        # For now, just return the structural rank if it matches
        if rank == n * m * p:
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            idx = np.arange(rank)
            i = (idx // (m * p)) % n
            j = (idx // p) % m
            k = idx % p
            U[i * m + j, idx] = 1
            V[j * p + k, idx] = 1
            W[k * n + i, idx] = 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # For ranks less than n*m*p, we would need to search
        # This is a placeholder for future search
        return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # Start with the structural rank decomposition (rank = n*m*p = 32)
    rank = n * m * p
    idx = np.arange(rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    i = (idx // (m * p)) % n
    j = (idx // p) % m
    k = idx % p
    U[i * m + j, idx] = 1
    V[j * p + k, idx] = 1
    W[k * n + i, idx] = 1
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try to find lower-rank decompositions
    # For 2x4x4, the known lower bounds are non-trivial.
    # The tensor rank of M_{2,4,4} is not known to be less than 32 in general,
    # but let's try a simple randomized search for small improvements.
    
    # Attempt randomized search for rank < best_rank
    for target_rank in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if time.time() - t0 > deadline - 30:
            break
        
        # Randomized initialization
        found = False
        for attempt in range(50):
            if time.time() - t0 > deadline - 10:
                break
            
            # Initialize with random small integers or half-integers
            U_rand = np.zeros((n * m, target_rank), dtype=float)
            V_rand = np.zeros((m * p, target_rank), dtype=float)
            W_rand = np.zeros((p * n, target_rank), dtype=float)
            
            # Seed with some structure
            for c in range(target_rank):
                # Randomly assign some non-zero entries
                for ii in range(n):
                    for jj in range(m):
                        if np.random.random() < 0.3:
                            U_rand[ii * m + jj, c] = np.random.choice([-1, 1])
                for jj in range(m):
                    for kk in range(p):
                        if np.random.random() < 0.3:
                            V_rand[jj * p + kk, c] = np.random.choice([-1, 1])
                for kk in range(p):
                    for ii in range(n):
                        if np.random.random() < 0.3:
                            W_rand[kk * n + ii, c] = np.random.choice([-1, 1])
            
            # Check if this random assignment works
            if _verify_factors(U_rand, V_rand, W_rand, n, m, p, tol=1e-6):
                # Found a valid decomposition!
                best_rank = target_rank
                best = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                found = True
                break
        
        if found:
            # Continue trying even lower ranks
            continue
        else:
            # If we can't find rank target_rank, break (assumption: lower ranks are harder)
            break
    
    return best
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    # Verify: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{row of C}
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    # Build C from factors
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
