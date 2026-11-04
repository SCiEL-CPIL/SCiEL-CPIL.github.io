"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 60

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U[0].__len__() if isinstance(U, list) else U.shape[1]
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check shapes
    if U_np.shape != (n*m, rank):
        return False
    if V_np.shape != (m*p, rank):
        return False
    if W_np.shape != (p*n, rank):
        return False
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if (i,j,k) is a valid triple (all in range), 0 otherwise.
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the rank-r decomposition gives:
    # (AB)_{i,k} = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not quite right. Let me think again.
    #
    # Standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor form: T_{i,j,k} where the "contraction" is over j.
    # But in the tensor decomposition for matrix multiplication, we have:
    # (AB)_{i,k} = sum_c U_{i, c} V_{k, c} ... no, that's not right either.
    #
    # Let me re-read the convention: W[k*n+i, r]
    # U has shape (n*m, rank): indexed by (i, j) -> i*m+j
    # V has shape (m*p, rank): indexed by (j, k) -> j*p+k  
    # W has shape (p*n, rank): indexed by (k, i) -> k*n+i
    #
    # The decomposition is: (AB)_{i,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But wait, j is summed over. So:
    # (AB)_{i,k} = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # This should equal sum_j A_{i,j} B_{j,k} for all A, B.
    # So we need: for all i, k, sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A_{i,j} B_{j,k} = sum_j A_{i,j} B_{j,k}
    # This means: sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,k}... no.
    #
    # Actually, the correct condition is that the bilinear form matches. Let me just verify
    # by checking that the computed (AB)_{i,k} matches the true product for random matrices.
    
    # Better approach: verify by checking the tensor identity directly.
    # The multiplication tensor T has entries T[i,j,k] = 1 (it's the identity-like tensor).
    # The rank-r decomposition satisfies: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's a trilinear form, but matrix multiplication is bilinear in A and B.
    
    # Let me just do the direct check with random matrices.
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        # Compute using decomposition
        AB_decomp = np.zeros((n, p))
        for c in range(rank):
            # For each c, we add a contribution
            # U[:, c] is a vector of length n*m, reshape to (n, m)
            # V[:, c] is a vector of length m*p, reshape to (m, p)
            # W[:, c] is a vector of length p*n, reshape to (p, n)
            Uc = U_np[:, c].reshape(n, m)
            Vc = V_np[:, c].reshape(m, p)
            Wc = W_np[:, c].reshape(p, n)
            # The contribution to (AB)_{i,k} from term c is:
            # sum_j Uc[i,j] * Vc[j,k] * Wc[k,i]
            # = (Uc @ Vc) * Wc element-wise... no
            # sum_j Uc[i,j] * Vc[j,k] = (Uc @ Vc)[i,k]
            # Then multiply by Wc[k,i] and sum over k? No, k is the output index.
            # Actually: contribution to (AB)_{i,k} = sum_j Uc[i,j] * Vc[j,k] * Wc[k,i]
            # This is (Uc @ Vc)_{i,k} * Wc[k,i]
            # So AB_decomp[i,k] += (Uc @ Vc)[i,k] * Wc[k,i]
            # But Wc is (p, n), so Wc[k,i] means Wc[k, i]
            UV = Uc @ Vc  # (n, p)
            # UV[i,k] * Wc[k,i]... but Wc[k,i] where k is row, i is col
            # Wc is (p, n), so Wc[k, i] is valid
            AB_decomp += UV * Wc.T  # Wc.T is (n, p), so Wc.T[i,k] = Wc[k,i]
        
        if not np.allclose(AB, AB_decomp, atol=tol, rtol=tol):
            return False
    return True

def _schoolbook():
    rank = RANK_SCHOOLBOOK
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

def _random_decomposition(rank, seed=None):
    """Generate a random rank-r decomposition and verify it."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Verify
    for _ in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB = A @ B
        AB_decomp = np.zeros((n, p))
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            UV = Uc @ Vc
            AB_decomp += UV * Wc.T
        if not np.allclose(AB, AB_decomp, atol=1e-7, rtol=1e-7):
            return None
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try to find lower rank decompositions using random search
    # For 2x5x6, the known minimum rank is at most 60 (schoolbook).
    # Let's try a few ranks below 60 with random initializations.
    
    for rank in range(best_rank - 1, max(20, best_rank - 30), -1):
        if time.time() > deadline - 30:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            result = _random_decomposition(rank, seed=attempt * 1000 + rank)
            if result is not None:
                best = result
                best_rank = rank
                found = True
                break
        if found:
            continue  # try even lower
        # If we didn't find at this rank, we can stop (unlikely to find lower)
        break
    
    return best
# EVOLVE-BLOCK-END
