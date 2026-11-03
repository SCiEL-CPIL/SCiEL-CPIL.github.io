import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 145  # seconds, leaving 5s margin for the 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The matrix multiplication tensor T for (n x m) * (m x p) -> (n x p)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Rank = minimum r such that T = sum_{r'=0}^{r-1} u_{r'} * v_{r'} * w_{r'}
    # where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    
    # We'll use a structured approach:
    # 1. First, try to find a low-rank decomposition using random initialization + gradient descent
    # 2. Use the fact that we can decompose into blocks
    
    # Known lower bound: R(n*m*p) >= max(n*m, m*p, p*n, (n*m*p)^{2/3})
    # For 3*4*5 = 60: lower bound is max(12, 20, 15, 60^{2/3}≈15.3) = 20
    
    # Let's try ranks from 20 upward, but be smart about it
    
    # Strategy: Use a Kronecker/structured decomposition if possible,
    # otherwise use random search with gradient descent
    
    # First, let's try a structured approach using Strassen-like decompositions
    
    # For 3x4x5, we can try:
    # - Decompose the 4 into 2+2 and use Strassen for 2x2 (rank 7 per 2x2x2)
    # - Or use the fact that 3x4x5 can be viewed as blocks
    
    # Let's try a direct approach: random initialization + Newton/gradient descent
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] = T (the MM tensor)"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        # Compute the tensor
        T_computed = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_computed += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        # The target tensor: T[i*m+j, j*p+k, k*n+i] = 1
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        diff = np.abs(T_computed - T_target)
        return diff.max() < tol, diff.max()
    
    def try_random_search(rank, max_iters=5000, lr=0.01, seed=42):
        """Try to find a rank-r decomposition via gradient descent"""
        if time.time() - start_time > TIME_LIMIT - 15:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        
        # Target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Gradient descent
        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 15:
                return None
            
            # Compute current tensor
            T_comp = np.zeros_like(T_target)
            for r_idx in range(rank):
                T_comp += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            # Error
            E = T_comp - T_target
            err = np.abs(E).max()
            
            if err < 1e-8:
                return U, V, W
            
            # Gradients
            # dU[:,r] = sum_{j,k} E[i*m+j, j*p+k, :] * V[j*p+k, r] * W[:, r]
            # This is expensive. Let's use a simpler update.
            
            # Actually, for the tensor decomposition, the gradients are:
            # dU[a, r] = sum_{b,c} E[a, b, c] * V[b, r] * W[c, r]
            # dV[b, r] = sum_{a,c} E[a, b, c] * U[a, r] * W[c, r]
            # dW[c, r] = sum_{a,b} E[a, b, c] * U[a, r] * V[b, r]
            
            # This is O(rank * n*m*m*p*p*n) which is very expensive for rank > 20
            # Let's use a different approach: alternating least squares or just try small ranks
            
            # For now, let's just do a simple check and break
            if it % 1000 == 0 and err > 1.0:
                # Restart with new initialization
                U = rng.randn(n*m, rank) * 0.1
                V = rng.randn(m*p, rank) * 0.1
                W = rng.randn(p*n, rank) * 0.1
        
        return None
    
    def try_als(rank, max_iters=200, tol=1e-6):
        """Alternating Least Squares for tensor decomposition"""
        if time.time() - start_time > TIME_LIMIT - 15:
            return None
        
        rng = np.random.RandomState(rank * 1000 + 42)
        
        # Target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Initialize
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 15:
                return None
            
            # Update U
            for r_idx in range(rank):
                # Fix V[:, r_idx] and W[:, r_idx], solve for U[:, r_idx]
                # T_target[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed r, the contribution is U[:, r] * (V[:, r] * W[:, r])
                # We want to minimize ||T_target - sum_r' U[:, r'] * V[:, r'] * W[:, r']||^2
                # This is not a simple linear solve for U[:, r] alone
                # Let's use a different approach: treat it as a matrix factorization
                
                # Actually, let's use the fact that for each r, we can solve:
                # Let M_r = V[:, r] * W[:, r] (outer product, shape m*p x p*n)
                # Then T_target = sum_r U[:, r] * M_r
                # Reshape: T_target as (n*m, m*p*p*n)
                # U * M_r flattened...
                pass
            
            # This is getting complicated. Let's try a different strategy.
            break
        
        return None
    
    # Better strategy: Use known decompositions
    # For 3x4x5, we can use:
    # - Schoolbook: rank 60
    # - Try to do better using structure
    
    # Let's try a block decomposition:
    # Split the 4 columns of B into two groups of 2
    # Then we have two 3x4x2 subproblems
    # Each 3x4x2 can be done with rank at most 3*4*2 = 24
    # Total: 48, which is better than 60
    
    # But can we do even better?
    # R(3x4x2) >= max(12, 8, 6, (24)^{2/3}≈8.3) = 12
    # So two of them: at least 24
    
    # Let's implement a rank-24 decomposition for 3x4x2
    # Actually, let's just use schoolbook for each part: rank 24 each, total 48
    
    # But we can do better! Let's use the fact that 3x4x2 = 3x(2+2)x2
    # Or use Strassen-like tricks
    
    # For now, let's implement the block decomposition for rank 48
    
    # Actually, let's try an even better approach:
    # Use the identity that R(n1*n2 x m x p) <= R(n1 x m x p) * n2 + R(n2 x m x p) * n1
    # This isn't directly helpful
    
    # Let's just implement a clean schoolbook and try to improve with random search for small ranks
    
    # First, return schoolbook as fallback
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook_decomposition()
    
    # Try to find a better decomposition
    # Use a structured approach: split into blocks
    
    # For 3x4x5:
    # Split the 5 columns into 4+1
    # 3x4x4 can potentially be done with rank < 48
    # 3x4x1 has rank at most 12, at least 12
    
    # For 3x4x4:
    # We can use the fact that 4=2+2
    # 3x4x4 = 3x(2+2)x(2+2) = 3x2x2 + 3x2x2 + 3x2x2 + 3x2x2
    # Each 3x2x2 has rank at most 12, at least 6
    # Total: 48, same as schoolbook for 3x4x4
    
    # But we can use Strassen for the 2x2 parts!
    # 3x2x2: treat as 3 copies of 2x2x2
    # Each 2x2x2 can be done with rank 7 (Strassen)
    # So 3x2x2 can be done with rank 3*7 = 21? No, that's not right.
    # 3x2x2 means A is 3x2, B is 2x2, C is 3x2
    # This is NOT 3 copies of 2x2x2. It's a single 3x2x2 tensor.
    # R(3x2x2) >= max(6, 4, 6, (12)^{2/3}≈5.2) = 6
    # Schoolbook: rank 12
    # Can we do rank 6? That would be perfect.
    
    # Let's try to find a rank-6 decomposition for 3x2x2
    # T[i*2+j, j*2+k, k*3+i] = 1 for i=0,1,2; j=0,1; k=0,1
    
    # If we can find R(3x2x2) = 6, then:
    # 3x4x4 = 3x(2+2)x(2+2)
    # We can decompose this as 4 sub-tensors of size 3x2x2
    # Each with rank 6, total rank 24
    # Plus 3x4x1 with rank 12
    # Total: 36
    
    # Let's try to find rank-6 for 3x2x2
    
    def find_3x2x2_decomposition():
        """Find a rank-6 decomposition for 3x2x2 matrix multiplication"""
        nn, mm, pp = 3, 2, 2
        rank = 6
        U = np.zeros((nn*mm, rank))
        V = np.zeros((mm*pp, rank))
        W = np.zeros((pp*nn, rank))
        
        # Target: T[i*2+j, j*2+k, k*3+i] = 1
        # Let's try a structured approach
        # A[i,j], B[j,k], C[i,k] = sum_j A[i,j]*B[j,k]
        # For 3x2x2, we need 6 multiplications
        
        # Try: 
        # P1 = A[0,0]*B[0,0]
        # P2 = A[0,1]*B[1,0]  
        # P3 = A[1,0]*B[0,1]
        # P4 = A[1,1]*B[1,1]
        # P5 = (A[2,0]+A[2,1])*B[0,0+1]... this is getting complex
        
        # Let's use a known result or try random search
        for seed in range(20):
            if time.time() - start_time > TIME_LIMIT - 10:
                return None
            rng = np.random.RandomState(seed)
            U_t = rng.randn(nn*mm, rank) * 0.5
            V_t = rng.randn(mm*pp, rank) * 0.5
            W_t = rng.randn(pp*nn, rank) * 0.5
            
            # Verify
            T_comp = np.zeros((nn*mm, mm*pp, pp*nn))
            for r in range(rank):
                T_comp += np.einsum('a,b,c->abc', U_t[:, r], V_t[:, r], W_t[:, r])
            
            T_target = np.zeros((nn*mm, mm*pp, pp*nn))
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        T_target[i*mm+j, j*pp+k, k*nn+i] = 1.0
            
            if np.abs(T_comp - T_target).max() < 1e-6:
                return U_t.tolist(), V_t.tolist(), W_t.tolist()
        
        return None
    
    # Try to find the 3x2x2 decomposition
    result_322 = find_3x2x2_decomposition()
    
    if result_322 and time.time() - start_time < TIME_LIMIT - 20:
        U_322, V_322, W_322 = result_322
        
        # Build 3x4x4 decomposition using 4 copies of 3x2x2
        # A is 3x4 = [A_left (3x2), A_right (3x2)]
        # B is 4x4 = [[B11, B12], [B21, B22]] where each is 2x2
        # C is 3x4 = [C_left (3x2), C_right (3x2)]
        # C_left = A_left * B11 + A_right * B21
        # C_right = A_left * B12 + A_right * B22
        
        # Each of the 4 products (A_left*B11, A_right*B21, A_left*B12, A_right*B22)
        # is a 3x2x2 problem with rank 6
        # Total rank: 24 for 3x4x4
        
        rank_344 = 24
        U_344 = np.zeros((3*4, rank_344))
        V_344 = np.zeros((4*4, rank_344))
        W_344 = np.zeros((4*3, rank_344))
        
        # Copy 1: A_left * B11 -> C_left
        for r in range(6):
            for i in range(3):
                for j in range(2):  # left half
                    U_344[i*4 + j, r] = U_322[i*2 + j][r]
            for j in range(2):
                for k in range(2):
                    V_344[j*4 + k, r] = V_322[j*2 + k][r]
            for k in range(2):
                for i in range(3):
                    W_344[k*3 + i, r] = W_322[k*3 + i][r]
        
        # Copy 2: A_right * B21 -> C_left
        for r in range(6):
            for i in range(3):
                for j in range(2, 4):  # right half
                    U_344[i*4 + j, r+6] = U_322[i*2 + (j-2)][r]
            for j in range(2):
                for k in range(2, 4):  # B21: rows 2-3, cols 0-1
                    V_344[j*4 + k, r+6] = V_322[(j+2)*2 + k][r]
            for k in range(2):
                for i in range(3):
                    W_344[k*3 + i, r+6] = W_322[k*3 + i][r]
        
        # Copy 3: A_left * B12 -> C_right
        for r in range(6):
            for i in range(3):
                for j in range(2):
                    U_344[i*4 + j, r+12] = U_322[i*2 + j][r]
            for j in range(2):
                for k in range(2, 4):  # B12: rows 0-1, cols 2-3
                    V_344[j*4 + k, r+12] = V_322[j*2 + (k-2)][r]
            for k in range(2, 4):
                for i in range(3):
                    W_344[k*3 + i, r+12] = W_322[(k-2)*3 + i][r]
        
        # Copy 4: A_right * B22 -> C_right
        for r in range(6):
            for i in range(3):
                for j in range(2, 4):
                    U_344[i*4 + j, r+18] = U_322[i*2 + (j-2)][r]
            for j in range(2):
                for k in range(2, 4):
                    V_344[j*4 + k, r+18] = V_322[(j+2)*2 + (k-2)][r]
            for k in range(2, 4):
                for i in range(3):
                    W_344[k*3 + i, r+18] = W_322[(k-2)*3 + i][r]
        
        # Now add 3x4x1 for the 5th column: rank 12
        rank_total = rank_344 + 12
        U_full = np.zeros((3*4, rank_total))
        V_full = np.zeros((4*5, rank_total))
        W_full = np.zeros((5*3, rank_total))
        
        U_full[:, :24] = U_344
        V_full[:, :24] = V_344
        W_full[:, :24] = W_344
        
        # Schoolbook for 3x4x1 (the 5th column)
        for idx in range(12):
            i = idx // 4
            j = idx % 4
            U_full[i*4 + j, 24 + idx] = 1
            V_full[j*5 + 4, 24 + idx] = 1
            W_full[4*3 + i, 24 + idx] = 1
        
        # Verify
        valid, max_err = verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist())
        if valid:
            best_U = U_full.tolist()
            best_V = V_full.tolist()
            best_W = W_full.tolist()
            best_rank = rank_total
    
    # Try even better: use rank-7 Strassen for 2x2x2
    # If we can find R(3x2x2) = 7 or less, we might improve
    
    # Actually, let's try rank 5 for 3x2x2
    def find_3x2x2_rank(r):
        for seed in range(10):
            if time.time() - start_time > TIME_LIMIT - 10:
                return None
            rng = np.random.RandomState(seed + 1000)
            U_t = rng.randn(6, r) * 0.5
            V_t = rng.randn(4, r) * 0.5
            W_t = rng.randn(6, r) * 0.5
            
            T_comp = np.zeros((6, 4, 6))
            for rr in range(r):
                T_comp += np.einsum('a,b,c->abc', U_t[:, rr], V_t[:, rr], W_t[:, rr])
            
            T_target = np.zeros((6, 4, 6))
            for i in range(3):
                for j in range(2):
                    for k in range(2):
                        T_target[i*2+j, j*2+k, k*3+i] = 1.0
            
            if np.abs(T_comp - T_target).max() < 1e-6:
                return U_t.tolist(), V_t.tolist(), W_t.tolist()
        return None
    
    # Try rank 5
    if time.time() - start_time < TIME_LIMIT - 30:
        result_322_r5 = find_3x2x2_rank(5)
        if result_322_r5:
            # Build with rank 5 per block: 4*5 = 20 for 3x4x4, +12 = 32 total
            U_322, V_322, W_322 = result_322_r5
            rank_344 = 20
            U_344 = np.zeros((3*4, rank_344))
            V_344 = np.zeros((4*4, rank_344))
            W_344 = np.zeros((4*3, rank_344))
            
            for copy_idx in range(4):
                r_off = copy_idx * 5
                for r in range(5):
                    if copy_idx == 0:
                        # A_left * B11 -> C_left
                        for i in range(3):
                            for j in range(2):
                                U_344[i*4 + j, r_off + r] = U_322[i*2 + j][r]
                        for j in range(2):
                            for k in range(2):
                                V_344[j*4 + k, r_off + r] = V_322[j*2 + k][r]
                        for k in range(2):
                            for i in range(3):
                                W_344[k*3 + i, r_off + r] = W_322[k*3 + i][r]
                    elif copy_idx == 1:
                        # A_right * B21 -> C_left
                        for i in range(3):
                            for j in range(2):
                                U_344[i*4 + j + 2, r_off + r] = U_322[i*2 + j][r]
                        for j in range(2):
                            for k in range(2):
                                V_344[(j+2)*4 + k, r_off + r] = V_322[j*2 + k][r]
                        for k in range(2):
                            for i in range(3):
                                W_344[k*3 + i, r_off + r] = W_322[k*3 + i][r]
                    elif copy_idx == 2:
                        # A_left * B12 -> C_right
                        for i in range(3):
                            for j in range(2):
                                U_344[i*4 + j, r_off + r] = U_322[i*2 + j][r]
                        for j in range(2):
                            for k in range(2):
                                V_344[j*4 + k + 2, r_off + r] = V_322[j*2 + k][r]
                        for k in range(2):
                            for i in range(3):
                                W_344[(k+2)*3 + i, r_off + r] = W_322[k*3 + i][r]
                    else:
                        # A_right * B22 -> C_right
                        for i in range(3):
                            for j in range(2):
                                U_344[i*4 + j + 2, r_off + r] = U_322[i*2 + j][r]
                        for j in range(2):
                            for k in range(2):
                                V_344[(j+2)*4 + k + 2, r_off + r] = V_322[j*2 + k][r]
                        for k in range(2):
                            for i in range(3):
                                W_344[(k+2)*3 + i, r_off + r] = W_322[k*3 + i][r]
            
            rank_total = rank_344 + 12
            U_full = np.zeros((3*4, rank_total))
            V_full = np.zeros((4*5, rank_total))
            W_full = np.zeros((5*3, rank_total))
            
            U_full[:, :20] = U_344
            V_full[:, :20] = V_344
            W_full[:, :20] = W_344
            
            for idx in range(12):
                i = idx // 4
                j = idx % 4
                U_full[i*4 + j, 20 + idx] = 1
                V_full[j*5 + 4, 20 + idx] = 1
                W_full[4*3 + i, 20 + idx] = 1
            
            valid, max_err = verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist())
            if valid:
                best_U = U_full.tolist()
                best_V = V_full.tolist()
                best_W = W_full.tolist()
                best_rank = rank_total
    
    # Convert to lists if needed
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
        best_V = best_V.tolist()
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
