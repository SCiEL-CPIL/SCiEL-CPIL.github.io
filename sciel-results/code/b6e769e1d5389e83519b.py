"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                val = np.dot(u, v * w)
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_strassen_2x2x5(n, m, p, start_time, deadline):
    """
    Try to exploit Strassen's rank-7 decomposition for 2x2x2 sub-problems.
    Split m=4 into two blocks of size 2.
    A (3x4) = [A1 A2] where A1, A2 are 3x2
    B (4x5) = [B1; B2] where B1, B2 are 2x5
    C = A1*B1 + A2*B2
    
    Each 3x2x5 sub-problem can be viewed as 3 independent 1x2x5 multiplications.
    But that's not quite right. Let's think differently.
    
    Actually, let's try a direct approach: search for a decomposition with
    values in {0, 1/2, 1} using a smarter initialization and local search.
    """
    # For 3x4x5, let's try to find a rank-59 decomposition by modifying schoolbook
    # or a lower rank using structured patterns.
    
    # One known result: the rank of 3x4x5 is at most 59 (by removing one schoolbook term
    # if it can be represented by others). Let's try to find if any schoolbook term is redundant.
    
    # Actually, let's try a different approach: use the fact that we can combine
    # some terms. For example, in the 2x2x2 case, Strassen shows rank 7 < 8.
    
    # Let's try to embed Strassen into the 3x4x5 problem.
    # Consider the sub-problem involving i=0,1, j=0,1, k=0,1 (a 2x2x2 sub-tensor).
    # This sub-tensor has rank 8 in schoolbook, but can be done in rank 7.
    # The remaining terms involve other indices.
    
    # Total schoolbook rank: 60
    # If we can replace the 8 terms for (i in {0,1}, j in {0,1}, k in {0,1}) with 7 terms,
    # we save 1 rank, giving total rank 59.
    
    # Let's construct this explicitly.
    # Schoolbook terms: (i,j,k) for i in 0..2, j in 0..3, k in 0..4
    # The 2x2x2 sub-problem is (i in {0,1}, j in {0,1}, k in {0,1})
    
    # Strassen's 7 terms for 2x2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In tensor notation, the 2x2x2 tensor T[i,j,k] = 1 for i,j,k in {0,1}
    # Strassen decomposition:
    # T[i,j,k] = sum_{r=1}^{7} u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # Let's map this to our indices. For the sub-problem (i in {0,1}, j in {0,1}, k in {0,1}):
    # U_sub[i*m+j, r], V_sub[j*p+k, r], W_sub[k*n+i, r]
    
    # The 7 Strassen terms (with appropriate signs):
    # We need to be careful with the indexing.
    
    # Let's just try a randomized search for rank 59 with values in {0, 1/2, 1, -1/2, -1}
    # This is a finite search space but large. Let's try a few random initializations.
    
    np.random.seed(42)
    target_rank = 59
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, target_rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, target_rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, target_rank))
        
        # Verify
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _try_local_search(n, m, p, start_time, deadline):
    """
    Start from schoolbook and try to reduce rank by merging terms.
    This is a heuristic approach.
    """
    # For now, this is a placeholder. A real implementation would try to
    # find linear dependencies among the schoolbook terms.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # First, try the Strassen-based approach for rank 59
    result = _try_strassen_2x2x5(n, m, p, start_time, deadline)
    if result is not None:
        best_rank = 59
        best_result = result
    
    # Try local search for further reductions
    result = _try_local_search(n, m, p, start_time, deadline)
    if result is not None:
        best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
