"""Rewritten program: structured search for low-rank decomposition of 4x4*4x9 matrix multiplication tensor."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses structured search with fast vectorized verification.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification using NumPy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Build the full tensor from decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        total = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)  # (n, m)
            Vc = V[:, c].reshape(m, p)  # (m, p)
            Wc = W[:, c].reshape(p, n)  # (p, n)
            # Uc[i,j] * Vc[j,k] * Wc[k,i] for all i,j,k
            for i in range(n):
                contrib = np.outer(Uc[i, :], Wc[:, i])  # (m, p), contrib[j,k] = Uc[i,j]*Wc[k,i]
                total[i, :, :] += contrib * Vc  # element-wise multiply by Vc[j,k]
        
        if np.max(np.abs(total - 1.0)) > tol:
            return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition (rank n*m*p)."""
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
        return U, V, W
    
    def try_deterministic_search(rank, max_attempts=500):
        """
        Deterministic combinatorial search for a rank decomposition.
        Uses structured patterns based on the tensor structure.
        """
        if time.time() > deadline - 20:
            return None
        
        # Strategy: Use the fact that we need T[i,j,k] = 1 for all i,j,k
        # Each rank-1 term contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # For a 4x4*4x9 multiplication, the schoolbook rank is 144.
        # We try to find a lower rank by sharing computation across k indices.
        
        # Key insight: For fixed (i,j), we need sum_k V[j*p+k,c]*W[k*n+i,c] to be related to U[i*m+j,c]
        # Actually, T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Let's try a systematic approach:
        # 1. Fix U to have a specific structure
        # 2. Solve for V and W
        
        # Approach: Use the fact that for each (i,j), the vector (V[j*p+0,c]*W[0*n+i,c], ..., V[j*p+(p-1),c]*W[(p-1)*n+i,c]) 
        # weighted by U[i*m+j,c] should sum to give 1 for each k.
        
        # Simpler approach: Try to find a decomposition where each term affects a small subset of indices.
        
        # Let's try a greedy approach:
        # Start with zero matrices, and for each (i,j,k) that hasn't been satisfied, add a rank-1 term.
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Track which (i,j,k) are satisfied
        residual = np.ones((n, m, p))
        
        for c in range(rank):
            if time.time() > deadline - 10:
                break
            
            # Find the (i,j,k) with the largest residual
            max_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
            i, j, k = max_idx
            
            # Create a rank-1 term that helps satisfy this (i,j,k)
            # Set U[i*m+j,c] = 1, V[j*p+k,c] = 1, W[k*n+i,c] = 1
            # This will add 1 to T[i,j,k] and potentially affect other (i',j',k')
            
            U[i*m+j, c] = 1
            V[j*p+k, c] = 1
            W[k*n+i, c] = 1
            
            # Update residual
            # The new term contributes U[i*m+j,c]*V[j'*p+k',c]*W[k'*n+i',c] to T[i',j',k']
            # For this term, it's non-zero only when i'=i, j'=j, k'=k (since U,V,W are sparse)
            # Actually, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
            # So the contribution to T[i',j',k'] is:
            # sum over (i',j',k') where U[i'*m+j',c]*V[j'*p+k',c]*W[k'*n+i',c] != 0
            # For this specific term, only (i,j,k) is affected
            residual[i, j, k] -= 1
        
        # Check if we've satisfied all entries
        if np.max(np.abs(residual)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    def try_structured_decomposition(rank):
        """
        Try a structured decomposition based on grouping the 9 output columns.
        """
        if time.time() > deadline - 20:
            return None
        
        # Split the 9 columns into groups: 3+3+3
        # For each group of 3 columns, we can use a 4x4*4x3 multiplication
        # Total schoolbook: 3 * 4*4*3 = 144 (same as full schoolbook)
        
        # But we can try to share computation between groups.
        
        # Alternative: Use a 4x4*4x9 decomposition with shared U factors.
        # For fixed i, the row i of A is multiplied by B to give row i of C.
        # C[i,:] = A[i,:] * B
        # This is a 1x4 * 4x9 = 1x9 multiplication.
        # Schoolbook rank for this: 4*9 = 36
        # Total for 4 rows: 4*36 = 144
        
        # Can we do better by sharing?
        # If A has structure (e.g., rows are linearly dependent), we might share.
        # But A is arbitrary, so we can't assume structure.
        
        # Let's try a different approach: use the fact that 9 = 3*3
        # Maybe we can use a 4x4*4x3 * 3x3 structure?
        
        # For now, let's try a random search with better structure.
        for attempt in range(200):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each column c: choose a random pattern
            for c in range(rank):
                # Choose which (i,j) pairs this term affects in U
                # Choose which (j,k) pairs this term affects in V
                # Choose which (k,i) pairs this term affects in W
                
                # Strategy: Make each term affect a small rectangle
                i_start = np.random.randint(0, n)
                j_start = np.random.randint(0, m)
                k_start = np.random.randint(0, p)
                
                i_len = np.random.randint(1, min(2, n - i_start) + 1)
                j_len = np.random.randint(1, min(2, m - j_start) + 1)
                k_len = np.random.randint(1, min(2, p - k_start) + 1)
                
                for i in range(i_start, i_start + i_len):
                    for j in range(j_start, j_start + j_len):
                        U[i*m+j, c] = np.random.choice([-1, 1])
                
                for j in range(j_start, j_start + j_len):
                    for k in range(k_start, k_start + k_len):
                        V[j*p+k, c] = np.random.choice([-1, 1])
                
                for k in range(k_start, k_start + k_len):
                    for i in range(i_start, i_start + i_len):
                        W[k*n+i, c] = np.random.choice([-1, 1])
            
            if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try to find lower rank decompositions
    best_result = None
    
    # Try ranks below schoolbook with various strategies
    for rank in range(140, schoolbook_rank + 1, -2):
        if time.time() > deadline - 20:
            break
        
        # Try deterministic search
        result = try_deterministic_search(rank)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, rank)
                break
        
        # Try structured decomposition
        result = try_structured_decomposition(rank)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
