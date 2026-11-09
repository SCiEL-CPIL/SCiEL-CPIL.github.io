"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Build the tensor T[i, j, k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Actually, the convention is:
        # (A B)[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: T[i, j, k] where the multiplication is sum over j
        # But the factor matrices are indexed differently.
        # U has shape (n*m, rank), so U[i*m+j, c]
        # V has shape (m*p, rank), so V[j*p+k, c]
        # W has shape (p*n, rank), so W[k*n+i, c]
        # The tensor entry T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And T[i, j, k] = 1 if this is the "middle" index j, else 0? 
        # Wait, the matrix multiplication tensor M[i, j, k] = delta_{j, j} which is just 1 for all i,j,k?
        # No. The tensor for matrix multiplication C = A * B is:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # So the tensor T[i, j, k] = 1 (it's the identity in the j-index contraction).
        # Actually, the tensor is T[i, j, k] = 1 for all i, j, k? No.
        # Let's re-read: "exactly decompose the matrix multiplication tensor"
        # The tensor is M[i, j, k] = 1 if we contract over j to get C[i,k].
        # So M[i, j, k] = 1 for all valid i, j, k.
        # Wait, that's not right either. The tensor M is defined such that
        # sum_j M[i, j, k] A[i, j] B[j, k] = C[i, k]
        # So M[i, j, k] = 1 for all i, j, k.
        # Therefore, we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
        
        # Let's verify this.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Start with schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb

    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Also rank >= n + m + p - 2 = 3 + 5 + 7 - 2 = 13 (but this is for square)
    # For rectangular, the lower bound is more complex.
    # Let's try ranks from 35 downward.
    
    # Actually, let's try a randomized search for lower rank.
    # We'll use a simple approach: start with schoolbook and try to merge columns.
    
    # Alternative: use the fact that for 3x5x7, the rank is known to be 35?
    # Let me check: the rank of matrix multiplication M_{n,m,p} is at least max(nm, mp, pn).
    # For 3,5,7: max(15, 35, 21) = 35.
    # It is also known that rank(M_{n,m,p}) = n*m*p for small dimensions, but for larger ones it can be less.
    # Actually, for 3x5x7, I believe the rank is 35.
    
    # Let's try to verify if rank 35 is achievable with a different decomposition.
    # If not, we'll stick with schoolbook (rank 105).
    
    # Wait, schoolbook is rank 105. Can we do better?
    # The lower bound is 35. Let's try to find a rank-35 decomposition.
    
    # One approach: use the "border rank" or "rank" of the tensor.
    # For matrix multiplication, the rank is known for many cases.
    # For 3x5x7, I'm not sure of the exact rank. Let's try a search.
    
    # Given time constraints, let's try a few random initializations and local search.
    
    def random_decomposition(rank, seed=None):
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        return U, V, W

    def local_search(U, V, W, max_iter=1000, tol=1e-8):
        """Try to reduce the error by adjusting entries."""
        rank = U.shape[1]
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Pick a random entry to update
            mat = random.choice([U, V, W])
            i = random.randint(0, mat.shape[0] - 1)
            j = random.randint(0, mat.shape[1] - 1)
            old_val = mat[i, j]
            # Try a small perturbation
            for delta in [0.1, -0.1, 0.01, -0.01, 0.5, -0.5]:
                mat[i, j] = old_val + delta
                # Check if error decreased
                err = 0.0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[ii * m + jj, c] * V[jj * p + kk, c] * W[kk * n + ii, c]
                            err += (s - 1.0) ** 2
                            if err > 1e-6:
                                break
                        if err > 1e-6:
                            break
                    if err > 1e-6:
                        break
                if err < 1e-6:
                    # Keep the change
                    break
                mat[i, j] = old_val
        return U, V, W

    # Try ranks from 35 to 104
    for target_rank in range(35, 105):
        if time.time() > deadline - 10:
            break
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            U, V, W = random_decomposition(target_rank, seed=42 + attempt)
            # Check if this random decomposition works (unlikely)
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        else:
            continue
        break

    # If we didn't find a lower rank, return schoolbook
    if best_rank == rank_sb:
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    else:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
