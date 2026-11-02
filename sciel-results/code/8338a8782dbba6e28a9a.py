import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n, m, p = 2, 4, 6
    
    # The multiplication tensor T has indices (i, j, k) where:
    # i in 0..n-1 (rows of A, rows of C)
    # j in 0..m-1 (cols of A, rows of B)
    # k in 0..p-1 (cols of B, cols of C)
    # T[i, j, k] = 1 if we are looking at the term A[i,j] * B[j,k] in C[i,k]
    
    # We want to find a rank-r decomposition:
    # T = sum_{r=1}^R U_r \otimes V_r \otimes W_r
    # Where:
    # U_r is a vector of size n*m (flattened A)
    # V_r is a vector of size m*p (flattened B)
    # W_r is a vector of size p*n (flattened C, with index k*n + i)
    
    # The condition is:
    # T[i, j, k] = sum_{r=1}^R U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    # This is equivalent to:
    # For all i, j, k:
    # sum_{r=1}^R U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    
    # Let's try to find a low-rank decomposition using a greedy approach or random initialization with refinement.
    # Given the small dimensions (8, 24, 12), we can attempt to solve for the factors directly if we fix one factor or use a specific structure.
    
    # However, finding the exact minimal rank is hard.
    # The current program uses rank 42.
    # Can we do better?
    # The trivial rank is n*m*p = 48.
    # Strassen gives 42.
    # Is 40 possible? Or 39?
    
    # Let's try a random search for a rank R < 42 decomposition.
    # We can formulate this as a system of polynomial equations.
    # Or, we can use a "tensor train" or "MPO" style search if we fix the structure, but the prompt asks for a general tensor decomposition.
    
    # A more effective approach for small tensors is to use the "subspace method" or "border rank" techniques, but those are complex.
    
    # Let's try a simple local search / gradient descent on the factors, minimizing the Frobenius norm of the difference between the reconstructed tensor and the target tensor.
    # Target tensor T is sparse (only 48 non-zeros, all 1s).
    
    # We will try ranks from 30 up to 41. If we find a valid integer/half-integer decomposition, we return it.
    # If not, we return the rank 42 solution.
    
    # To speed up, we can use the fact that the tensor is "structured".
    # Actually, let's just try to find ANY decomposition of rank R.
    # Since the entries must be integer/half-integer, we can round the solution from a real-valued optimization.
    
    # Let's define the target tensor as a 3D array.
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # We want to find U, V, W such that:
    # T[i, j, k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
    
    # Let's try a rank R decomposition.
    # We can fix W to be an identity-like structure or use random initialization.
    
    # Strategy:
    # 1. Try R = 30, 32, 34, 36, 38, 40.
    # 2. For each R, initialize U, V, W randomly.
    # 3. Use a few iterations of a simple alternating minimization (fix two, solve for the third).
    # 4. Check if the result is close to integer/half-integer.
    # 5. If yes, verify exactly.
    
    # This might be too slow or unreliable.
    # Let's stick to the known rank 42 solution but see if we can optimize it.
    # The prompt says "investigate improvements within its family".
    # The family is "Strassen-based block decomposition".
    
    # Is there a way to share factors between blocks?
    # In the current solution, each 2x2x2 block is independent.
    # If we could combine blocks, we might reduce rank.
    
    # Let's try a rank 40 solution by guessing a structure.
    # Or, let's just return the rank 42 solution as it is known to work and is fast.
    # The goal is to *improve* the score, which means finding a lower rank.
    # If I can't find a lower rank quickly, I should return the best known (42).
    
    # Let's try a very fast heuristic:
    # Can we reduce the rank by 2?
    # There are known results that for 2x4x6, the rank might be 40 or 39.
    # Without a search algorithm that runs in < 1s, it's hard to find.
    
    # Let's implement a quick random search for R=40.
    # If it fails, return R=42.
    
    def generate_strassen_solution():
        rank = 42
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        def add_strassen_block(a_idx, b_idx, c_idx, start_col):
            a, b, c, d = a_idx
            e, f, g, h = b_idx
            c11, c12, c21, c22 = c_idx
            
            U[a][start_col] += 1; U[d][start_col] += 1
            V[e][start_col] += 1; V[h][start_col] += 1
            W[c11][start_col] += 1; W[c22][start_col] += 1
            start_col += 1
            
            U[c][start_col] += 1; U[d][start_col] += 1
            V[e][start_col] += 1
            W[c21][start_col] += 1; W[c22][start_col] -= 1
            start_col += 1
            
            U[a][start_col] += 1
            V[f][start_col] += 1; V[h][start_col] -= 1
            W[c12][start_col] += 1; W[c22][start_col] += 1
            start_col += 1
            
            U[d][start_col] += 1
            V[g][start_col] += 1; V[e][start_col] -= 1
            W[c11][start_col] += 1; W[c21][start_col] += 1
            start_col += 1
            
            U[a][start_col] += 1; U[b][start_col] += 1
            V[h][start_col] += 1
            W[c11][start_col] -= 1; W[c12][start_col] += 1
            start_col += 1
            
            U[c][start_col] += 1; U[a][start_col] -= 1
            V[e][start_col] += 1; V[f][start_col] += 1
            W[c22][start_col] += 1
            start_col += 1
            
            U[b][start_col] += 1; U[d][start_col] -= 1
            V[g][start_col] += 1; V[h][start_col] += 1
            W[c11][start_col] += 1
            start_col += 1
            
            return start_col

        for k in range(3):
            A_L_idx = [0, 1, 4, 5]
            A_R_idx = [2, 3, 6, 7]
            B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
            B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
            C_idx = [4*k, 4*k+2, 4*k+1, 4*k+3]
            
            col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
            col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a better rank
    # We will try R=40.
    # This is a heuristic search.
    
    # If we have time, we can try a simple random search.
    # But given the 120s limit and the complexity, it is risky.
    
    # Let's just return the rank 42 solution for now, as it is guaranteed to work and is fast.
    # The "improvement" might be in the code structure or verifying the solution.
    
    # Wait, the prompt says "investigate improvements within its family".
    # Maybe we can merge some of the Strassen terms?
    # No, they are independent blocks.
    
    # Let's return the rank 42 solution.
    
    if time.time() > deadline:
         return generate_strassen_solution()
         
    return generate_strassen_solution()

def verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    # T[i, j, k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
    # We check if the reconstructed tensor matches the target (all 1s).
    # To be efficient, we can compute the tensor contraction.
    # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W) is not directly possible because indices are flattened.
    
    # Let's loop over i, j, k.
    # n=2, m=4, p=6 -> 48 checks.
    # R is likely < 50.
    # 48 * 50 = 2400 operations. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for r in range(len(U[0])):
                    s += U[r][u_idx] * V[r][v_idx] * W[r][w_idx]
                if s != 1:
                    return False
    return True

def search_rank_r(R, n, m, p, max_iters=100, seed=42):
    """
    Attempt to find a rank R decomposition using random initialization and alternating least squares.
    """
    np.random.seed(seed)
    
    # Initialize U, V, W with random values
    # We use small integers or half-integers to encourage sparsity/structure?
    # Random normal is standard for ALS.
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # Normalize to prevent blowup
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    # Target tensor T[i,j,k] = 1
    # We want to minimize || T - sum_r U_r V_r W_r ||_F^2
    
    # Alternating Least Squares:
    # Fix V, W -> Solve for U
    # Fix U, W -> Solve for V
    # Fix U, V -> Solve for W
    
    # The equation for U[r, i*m+j] is:
    # sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
    # This is linear in U if V, W are fixed.
    # Let A_U be a matrix where each row corresponds to a (i,j,k) triplet?
    # No, it's better to view it as:
    # For fixed r, the contribution to T[i,j,k] is U[r, i*m+j] V[r, j*p+k] W[r, k*n+i].
    # Summing over r gives T[i,j,k].
    
    # This is a linear system for U entries if we group by r?
    # Actually, for a fixed r, U[:, r] is a vector.
    # Let's solve for U[:, r] given V[:, r] and W[:, r] and the contributions from other r's?
    # No, ALS usually solves for one mode at a time, treating the other modes as fixed matrices.
    
    # Let's reshape U to (n, m, R), V to (m, p, R), W to (p, n, R).
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Fix V and W. We want to find U.
    # For each i, j, the equation is:
    # sum_r U[i,j,r] * (V[j, :, r] * W[:, i, r]) = 1 (for all k? No, for each k independently? No, T[i,j,k] is the target)
    # Actually, T[i,j,k] = 1 for all i,j,k.
    # So for fixed i, j, we have p equations (one for each k) and R unknowns (U[i,j,0..R-1]).
    # If R > p, we have an underdetermined system.
    # Let's form the matrix M_{ij} of size p x R.
    # M_{ij}[k, r] = V[j, k, r] * W[k, i, r]
    # We want M_{ij} @ U[i, j, :] = 1 (vector of ones of length p)
    
    # We can solve this least squares problem for each (i, j).
    
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                # Build M: p x R
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V_mat[j, k, :] * W_mat[k, i, :]
                # Solve M @ x = 1
                # x = U[i, j, :]
                try:
                    U[i, j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # Build M: n x R
                # Equation: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
                # Fix U, W. Solve for V[j, k, :].
                # M[i, r] = U[i, j, r] * W[k, i, r]
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U_mat[i, j, :] * W_mat[k, i, :]
                try:
                    V[j, k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # Build M: m x R
                # Equation: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
                # Fix U, V. Solve for W[k, i, :].
                # M[j, r] = U[i, j, r] * V[j, k, r]
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U_mat[i, j, :] * V_mat[j, k, :]
                try:
                    W[k, i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence / validity
        # Reshape back to flat
        U_flat = U_mat.reshape(n * m, R)
        V_flat = V_mat.reshape(m * p, R)
        W_flat = W_mat.reshape(p * n, R)
        
        # Check if close to integers/half-integers
        # If yes, round and verify
        U_rounded = np.round(U_flat * 2) / 2
        V_rounded = np.round(V_flat * 2) / 2
        W_rounded = np.round(W_flat * 2) / 2
        
        if np.allclose(U_flat, U_rounded, atol=1e-6) and np.allclose(V_flat, V_rounded, atol=1e-6) and np.allclose(W_flat, W_rounded, atol=1e-6):
             if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                 return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        if time.time() > deadline:
            return None

    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n, m, p = 2, 4, 6
    
    # Define target tensor T (all 1s) implicitly in verification
    
    # Try to find a rank 40 decomposition
    # We try a few seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        res = search_rank_r(40, n, m, p, max_iters=50, seed=seed)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Try rank 39
    for seed in range(10):
        if time.time() > deadline:
            break
        res = search_rank_r(39, n, m, p, max_iters=50, seed=seed)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}

    # Fallback to rank 42
    def generate_strassen_solution():
        rank = 42
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        def add_strassen_block(a_idx, b_idx, c_idx, start_col):
            a, b, c, d = a_idx
            e, f, g, h = b_idx
            c11, c12, c21, c22 = c_idx
            
            U[a][start_col] += 1; U[d][start_col] += 1
            V[e][start_col] += 1; V[h][start_col] += 1
            W[c11][start_col] += 1; W[c22][start_col] += 1
            start_col += 1
            
            U[c][start_col] += 1; U[d][start_col] += 1
            V[e][start_col] += 1
            W[c21][start_col] += 1; W[c22][start_col] -= 1
            start_col += 1
            
            U[a][start_col] += 1
            V[f][start_col] += 1; V[h][start_col] -= 1
            W[c12][start_col] += 1; W[c22][start_col] += 1
            start_col += 1
            
            U[d][start_col] += 1
            V[g][start_col] += 1; V[e][start_col] -= 1
            W[c11][start_col] += 1; W[c21][start_col] += 1
            start_col += 1
            
            U[a][start_col] += 1; U[b][start_col] += 1
            V[h][start_col] += 1
            W[c11][start_col] -= 1; W[c12][start_col] += 1
            start_col += 1
            
            U[c][start_col] += 1; U[a][start_col] -= 1
            V[e][start_col] += 1; V[f][start_col] += 1
            W[c22][start_col] += 1
            start_col += 1
            
            U[b][start_col] += 1; U[d][start_col] -= 1
            V[g][start_col] += 1; V[h][start_col] += 1
            W[c11][start_col] += 1
            start_col += 1
            
            return start_col

        for k in range(3):
            A_L_idx = [0, 1, 4, 5]
            A_R_idx = [2, 3, 6, 7]
            B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
            B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
            C_idx = [4*k, 4*k+2, 4*k+1, 4*k+3]
            
            col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
            col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() > deadline:
         return generate_strassen_solution()
         
    return generate_strassen_solution()
