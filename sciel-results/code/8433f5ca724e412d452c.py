import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build the target tensor T
    # T[u, v, w] = 1 if there exist i, j, k such that:
    # u = i*m + j, v = j*p + k, w = k*n + i
    # And the value is 1 (since A[i,j]*B[j,k] contributes to C[k,i]? No, C[i,k] usually.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Standard matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T_{i,j,k} = 1.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i  <-- This is C[k,i] in row-major? Or C[i,k]?
    # If W index is k*n + i, and we sum U[i*m+j, l] V[j*p+k, l] W[k*n+i, l],
    # then the result at (k, i) is sum_j ... which is (B^T A)^T? 
    # Let's check the schoolbook in the seed:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    # Sum over l: U V W -> sum over j (since i,k fixed, j varies? No, l varies).
    # For a fixed (i,j,k), there is exactly one l where all three are 1.
    # So T[u,v,w] = 1 for the tuple corresponding to (i,j,k).
    # The resulting tensor S[u,v,w] = sum_l U[u,l]V[v,l]W[w,l].
    # We want S[u,v,w] = T[u,v,w].
    
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
                
    # We want to find U, V, W of shape (9, r)
    # such that sum_l U[:,l] V[:,l] W[:,l] == T
    
    # Strategy:
    # 1. Try rank 23, 24, 25, 26.
    # 2. For each rank, try random initializations of U, V.
    # 3. Solve for W using least squares.
    # 4. Check if W is close to integers/half-integers.
    # 5. If yes, verify and return.
    
    # To make this faster, we pre-compute the structure.
    # For a fixed w, we solve A_w x = b_w.
    # A_w is (81 x r). A_w[idx, l] = U[u, l] * V[v, l].
    # b_w[idx] = T[u, v, w].
    
    # We can vectorize the check for integer/half-integer.
    
    def solve_and_check(r, U, V):
        """
        Given U, V of shape (9, r), solve for W.
        Returns W if valid, else None.
        """
        W = np.zeros((p*n, r))
        # Pre-compute A matrices for all w? 
        # A_w depends on U, V.
        # A_w is 81 x r.
        # We can build a big matrix? No, different w have different b.
        # But A_w is the same for all w? 
        # A_w[idx, l] = U[u, l] * V[v, l].
        # idx = u * (m*p) + v.
        # This A matrix is independent of w!
        # Let A be (81, r).
        A = np.zeros((n*m * m*p, r))
        for u in range(n*m):
            for v in range(m*p):
                idx = u * (m*p) + v
                A[idx, :] = U[u, :] * V[v, :]
                
        # Now for each w, we solve A x = b_w.
        # b_w is a vector of length 81.
        # b_w[idx] = T[u, v, w].
        
        # We can stack all b_w into a matrix B of shape (81, 9) where column w is b_w.
        B = np.zeros((n*m * m*p, p*n))
        for u in range(n*m):
            for v in range(m*p):
                idx = u * (m*p) + v
                # T[u, v, w] for all w
                # T is (9, 9, 9)
                B[idx, :] = T[u, v, :]
                
        # We want to solve A X = B, where X is (r, 9).
        # X[:, w] is W[w, :].
        try:
            # Use least squares
            X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        # Check residuals
        if residuals is not None and len(residuals) > 0:
            if residuals[0] > 1e-9:
                return None
        else:
            # Check if A @ X == B
            if not np.allclose(A @ X, B, atol=1e-9):
                return None
                
        # X is (r, 9). We want W to be (9, r).
        # W[w, l] = X[l, w].
        W = X.T
        
        # Check if W entries are close to integers or half-integers.
        # 2 * W should be close to integers.
        W_scaled = 2.0 * W
        W_rounded = np.round(W_scaled)
        
        if np.max(np.abs(W_scaled - W_rounded)) > 1e-9:
            return None
            
        # Also check U and V? The problem says "integer/half-integer entries".
        # We generated U, V with integers, so they are fine.
        # We snapped W to half-integers.
        W = W_rounded / 2.0
        
        # Final verification
        # Reconstruct T_rec
        # T_rec[u, v, w] = sum_l U[u, l] V[v, l] W[w, l]
        # We can compute this using the matrices.
        # T_rec = sum_l np.outer(np.outer(U[:,l], V[:,l]), W[:,l])? No.
        # T_rec is 3D.
        # We can check:
        # For each l, add U[:,l] * V[:,l] * W[:,l] to T_rec?
        # T_rec[u, v, w] = sum_l (U[u, l] * V[v, l] * W[w, l])
        
        # Efficient check:
        # Compute M = U @ V.T  (9x9)? No.
        # Let's just loop over l, it's small.
        T_rec = np.zeros((n*m, m*p, p*n), dtype=float)
        for l in range(r):
            # T_rec += np.einsum('i,j,k->ijk', U[:,l], V[:,l], W[:,l])
            # Or:
            T_rec += np.einsum('i,j->ij', U[:,l], V[:,l])[:, :, np.newaxis] * W[:,l][np.newaxis, np.newaxis, :]
            # Wait, np.einsum('i,j->ij', U[:,l], V[:,l]) gives (9,9).
            # Multiply by W[:,l] (9,) to get (9,9,9).
            
        if not np.allclose(T_rec, T, atol=1e-9):
            return None
            
        return W

    # Search
    random.seed(42)
    
    # Try ranks from 23 to 26
    for r in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
            
        # Number of tries
        # Rank 23 is hardest. Rank 26 is easier.
        # We will do many tries.
        max_tries = 1000
        for _ in range(max_tries):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            # Entries from {-1, 0, 1}
            # Sparse to help?
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Random sparse
            for i in range(n*m):
                for l in range(r):
                    if random.random() < 0.3:  # Sparsity
                        U[i, l] = random.choice([-1, 1])
                        
            for i in range(m*p):
                for l in range(r):
                    if random.random() < 0.3:
                        V[i, l] = random.choice([-1, 1])
                        
            W = solve_and_check(r, U, V)
            
            if W is not None:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

    # Fallback: Schoolbook rank 27
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
# EVOLVE-BLOCK-END
