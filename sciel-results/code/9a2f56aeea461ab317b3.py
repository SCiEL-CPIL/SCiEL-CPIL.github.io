"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 72
    
    # Build the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The multiplication tensor has entries T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # Actually, the tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # T is a 3-tensor of shape (n, m, p) where T[i,j,k] = 1 for all valid i,j,k
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the convention.
    
    # W[k*n+i, r] - so W is indexed by (k, i) flattened to k*n+i
    # U is (n*m, rank) - indexed by (i, j) flattened to i*m+j
    # V is (m*p, rank) - indexed by (j, k) flattened to j*p+k
    
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is the standard matrix multiplication tensor.
    
    # Known lower bounds: For 3x3x3, the rank is between 19 and 23 (Laderman is 23, Strassen is 7 for 2x2x2)
    # For 3x3x8, we can use block structure.
    
    # Let's try a structured approach. Since p=8, we can think of this as 8 separate 3x3 multiplications
    # but sharing the A matrix. Actually, it's one A (3x3) times 8 different B columns (3x8).
    
    # A better approach: Use the fact that we can decompose the 8-dimensional output space.
    # Let's try random rank minimization with verification.
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Efficient check: For each r, the outer product contribution
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape for efficient computation
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # For each r:
        #   U_r reshaped to (n, m)
        #   V_r reshaped to (m, p) 
        #   W_r reshaped to (p, n)
        #   Contribution to T[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
        
        total = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            # total[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
            for i in range(n):
                for j in range(m):
                    total[i, j, :] += Ur[i, j] * Vr[j, :] * Wr[:, i]
        
        # Check if total is all ones
        if not np.allclose(total, 1.0, atol=1e-9):
            return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using tensor operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each r, compute the contribution
        # Use einsum-like approach
        total = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            # total[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
            # This is: for each i, Ur[i,:] is a row, for each j, Vr[j,:] 
            # Let's do it differently:
            # temp[i,j,k] = Ur[i,j] * Vr[j,k] * Wr[k,i]
            # = (Ur[i,j] * Wr[i]) outer with Vr[j,k]... no
            # For fixed i: sum over j,k: Ur[i,j]*Vr[j,k]*Wr[k,i]
            # Let's just loop i
            for i in range(n):
                # Ur[i, :] has shape (m,)
                # Wr[i] is a scalar
                # For each j: Ur[i,j] * Wr[i] * Vr[j, :]
                total[i, :, :] += np.outer(Ur[i, :] * Wr[i], np.ones(p)) * Vr  # This isn't right
                # Actually: total[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                # For fixed i: total[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                # = (Ur[i,:] * Wr[i]) ... no, Wr[k,i] depends on k
                # Let me just do the triple loop for correctness
                for j in range(m):
                    total[i, j, :] += Ur[i, j] * Vr[j, :] * Wr[:, i]
        
        if not np.allclose(total, 1.0, atol=1e-9):
            return False
        return True
    
    def verify_fastest(U, V, W, rank):
        """Use vectorized operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        total = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            # total[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
            for i in range(n):
                # total[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                # For fixed i, this is: sum_j Ur[i,j] * Vr[j,k] * Wr[k,i] for each k
                # = (Ur[i,:] @ Vr) * Wr[:, i]  ... no
                # Let A_i = Ur[i, :] (shape m), then (A_i @ Vr) has shape p
                # Then total[i, :, k] = sum_j Ur[i,j]*Vr[j,k] * Wr[k,i]
                # = (Ur[i,:] @ Vr)[k] * Wr[k,i]
                # So total[i, :, :] = np.outer(np.ones(m), (Ur[i,:] @ Vr) * Wr[:, i])
                # No, that's not right either.
                # total[i, j, k] = Ur[i,j] * Vr[j,k] * Wr[k,i]
                # For fixed i: total[i, :, :] is a m x p matrix where
                # element (j,k) = Ur[i,j] * Vr[j,k] * Wr[k,i]
                # = diag(Ur[i,:]) @ Vr * diag(Wr[:,i])  ... in matrix form
                # Actually: total[i] = np.outer(Ur[i, :], np.ones(p)) * Vr * np.outer(np.ones(m), Wr[:, i])
                # = np.outer(Ur[i, :], Wr[:, i]) * Vr  ... no, that gives (j,k) -> Ur[i,j]*Wr[k,i]*Vr[j,k]? 
                # np.outer(Ur[i,:], Wr[:,i]) has shape (m,p) with element (j,k) = Ur[i,j]*Wr[k,i]
                # Multiply elementwise by Vr: gives Ur[i,j]*Wr[k,i]*Vr[j,k] = Ur[i,j]*Vr[j,k]*Wr[k,i]
                # Yes!
                total[i, :, :] += np.outer(Ur[i, :], Wr[:, i]) * Vr
        
        if not np.allclose(total, 1.0, atol=1e-9):
            return False
        return True
    
    # Start with schoolbook
    best_rank = rank_schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank using random initialization and optimization
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try various ranks from low to high
    # Lower bound: For matrix multiplication, rank >= max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Actually the lower bound is more subtle. For n x m x p, rank >= max(nm, mp, pn) is not always true.
    # A known lower bound is rank >= (nm + mp + pn - ... ) but let's just try empirically.
    
    # Let's try rank from 24 upward
    random.seed(42)
    
    for target_rank in range(24, min(best_rank + 1, 40)):
        if time.time() > deadline - 10:
            break
        
        attempts = 0
        max_attempts = 500 if target_rank <= 30 else 200
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple gradient-free approach: use the fact that T[i,j,k]=1
            # We can try to solve this as a system of equations
            # For each (i,j,k), we have: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a non-linear system. Let's try a different approach.
            # Since all entries of T are 1, we can try structured solutions.
            
            # Try: U[i*m+j, r] = a_r, V[j*p+k, r] = b_r, W[k*n+i, r] = c_r
            # Then sum_r a_r * b_r * c_r = 1 for all i,j,k
            # This requires sum_r a_r*b_r*c_r = 1, which is one equation.
            # But we need this to hold for all i,j,k, so if U,V,W are constant across i,j,k,
            # then we just need sum_r a_r*b_r*c_r = 1.
            # This gives rank 1 if a*b*c = 1 for some r!
            # Wait: if U[i*m+j, r] = a_r for all i,j, V[j*p+k, r] = b_r for all j,k, W[k*n+i, r] = c_r for all k,i
            # Then T[i,j,k] = sum_r a_r * b_r * c_r
            # We need this to equal 1. So pick a_0=1, b_0=1, c_0=1, rest zero. Rank 1!
            # But wait, is this valid? Let me check the indexing.
            # U has shape (n*m, rank). U[i*m+j, r] should depend on i,j.
            # If U[i*m+j, r] = a_r (independent of i,j), V[j*p+k, r] = b_r (independent of j,k), 
            # W[k*n+i, r] = c_r (independent of k,i)
            # Then sum_r a_r * b_r * c_r must equal 1.
            # With rank 1: a_0*b_0*c_0 = 1. Set a_0=1, b_0=1, c_0=1.
            # Let's verify!
            
            if target_rank == 1:
                U = np.zeros((n * m, 1))
                V = np.zeros((m * p, 1))
                W = np.zeros((p * n, 1))
                U[:, 0] = 1
                V[:, 0] = 1
                W[:, 0] = 1
                if verify_fastest(U, V, W, 1):
                    best_rank = 1
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            
            # For higher ranks, let's try a more structured approach
            # The issue is that rank 1 might not work because the tensor might not be separable
            # Let me re-examine: T[i,j,k] = 1 for all i,j,k
            # Is T separable? T[i,j,k] = f(i,j) * g(j,k) * h(k,i)?
            # If f(i,j) = 1, g(j,k) = 1, h(k,i) = 1, then T[i,j,k] = 1. Yes!
            # So rank 1 should work!
            
            # Wait, but the schoolbook decomposition has rank n*m*p = 72.
            # Why would rank 1 work? Let me double-check the tensor definition.
            
            # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor T is defined such that C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
            # So T[i,j,k] = 1 if the indices are valid, 0 otherwise.
            # In our case, all indices are valid, so T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8).
            
            # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # If U[i*m+j, r] = 1 for all i,j (for r=0), V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i
            # Then T[i,j,k] = 1*1*1 = 1. Correct!
            
            # So rank 1 works! Let's verify this is correct.
            
    # Let's just use rank 1 and verify
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    if verify_fastest(U, V, W, 1):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
