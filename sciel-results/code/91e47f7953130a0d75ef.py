"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.
    Uses the Kruskal CP decomposition with rational reconstruction.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    N = n * m  # 12
    M = m * p  # 28
    P = p * n  # 21
    
    # Build the matrix multiplication tensor T[i,j,k] = delta_{i,k} for appropriate indices
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the multiplication uses this term, else 0
    # More precisely, T is the structure tensor of matrix multiplication:
    # T[i, j, k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k]
    # But that's just the standard tensor where T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, no. The matrix multiplication tensor has entries:
    # T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # This is always 1 for valid indices, so T is just the all-ones tensor?
    
    # Let me reconsider. The matrix multiplication tensor T^{n,m,p} is defined such that:
    # C = A * B where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has three modes: i (size n), j (size m), k (size p)
    # T[i,j,k] = 1 if this is a valid multiplication term
    # For standard matrix multiplication, ALL terms are valid, so T[i,j,k] = 1 for all i,j,k
    
    # But that's not right either. The rank of the all-ones tensor of size n*m*p is 1.
    # That can't be the matrix multiplication tensor.
    
    # Let me think more carefully. The matrix multiplication tensor is:
    # T[i, j, k] where i indexes rows of A (and C), j indexes columns of A (and rows of B), 
    # k indexes columns of B (and C)
    # The entry T[i,j,k] = 1 means that A[i,j] * B[j,k] contributes to C[i,k]
    # This is true for ALL valid (i,j,k), so T is indeed the all-ones tensor.
    
    # But wait, the rank of the all-ones tensor of dimensions n x m x p is 1, which is clearly wrong.
    # The issue is that the tensor must be viewed differently.
    
    # Actually, I think the correct formulation is:
    # The matrix multiplication tensor has entries T[i,j,k] where:
    # - i ranges over 0..n-1 (row index of A and C)
    # - j ranges over 0..m-1 (column index of A and row index of B)  
    # - k ranges over 0..p-1 (column index of B and C)
    # And T[i,j,k] = 1 for all valid indices.
    
    # The rank of this tensor is the minimum number of rank-1 tensors needed to represent it.
    # A rank-1 tensor has the form u_i * v_j * w_k.
    # So we need to find rank r and vectors u_c, v_c, w_c for c=1..r such that:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k] for all i,j,k
    
    # For the all-ones tensor, this is trivially satisfied with r=1: u[i]=1, v[j]=1, w[k]=1.
    # But that's not the matrix multiplication tensor!
    
    # I think I'm confusing myself. Let me look at this differently.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    
    # The correct formulation: We want to compute C = A * B where A is n x m and B is m x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The tensor T has entries T[i,j,k] which is 1 if the product A[i,j]*B[j,k] is part of computing C[i,k].
    # This is ALWAYS 1 for valid indices, so T[i,j,k] = 1.
    
    # But then the rank is 1, which is clearly wrong. The known rank of 3x4x7 matrix multiplication 
    # is much larger than 1.
    
    # I think the issue is that the tensor is not just about which products are formed, but about 
    # how they combine. Let me reconsider.
    
    # Actually, I believe the matrix multiplication tensor is defined as follows:
    # T is a 3-tensor with dimensions (n*m, m*p, p*n) or similar, where each entry corresponds to 
    # a specific multiplication term.
    
    # Let me use the standard definition: The matrix multiplication tensor <n,m,p> has:
    # - First mode: n*m (indexing (i,j) pairs for A)
    # - Second mode: m*p (indexing (j,k) pairs for B)  
    # - Third mode: p*n (indexing (k,i) pairs for C, using trace convention W[k*n+i])
    
    # And T[(i,j), (j',k), (k',i')] = 1 if j=j' and i=i' and k=k', else 0.
    
    # So T is a 3-tensor of size (n*m) x (m*p) x (p*n) where:
    # T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j,k), c corresponds to (k,i)
    # and the indices match: the j in a equals the j in b, the k in b equals the k in c, 
    # and the i in c equals the i in a.
    
    # In other words, T is the "structure tensor" of matrix multiplication.
    
    # For n=3, m=4, p=7:
    # N = n*m = 12
    # M = m*p = 28
    # P = p*n = 21
    
    # T[a, b, c] = 1 if:
    #   a = i*m + j (0 <= i < n, 0 <= j < m)
    #   b = j'*p + k (0 <= j' < m, 0 <= k < p)
    #   c = k'*n + i' (0 <= k' < p, 0 <= i' < n)
    #   and j = j', k = k', i = i'
    
    # So T has exactly n*m*p = 84 non-zero entries (all 1s).
    
    # We want to find the minimum rank r such that:
    # T[a, b, c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    
    # where U is N x r, V is M x r, W is P x r.
    
    # This is the CP decomposition of the matrix multiplication tensor.
    
    # The schoolbook decomposition has rank n*m*p = 84.
    # We want to find if there's a lower rank decomposition.
    
    # For 3x4x7, the known minimum rank is not well-established, but it's known to be less than 84.
    
    # Let's try to find a rank-r decomposition using optimization.
    
    # Strategy: Try ranks from some lower bound up to 84, and for each rank, try to find 
    # a decomposition using alternating least squares or similar.
    
    # For exactness, we'll use rational reconstruction after converging to a solution.
    
    # Lower bound: The rank must be at least max(N, M, P) = 28? No, that's not a valid lower bound.
    # A better lower bound: The rank must be at least the size of the largest "slice" or something.
    # Actually, a simple lower bound is max(N, M, P) / something... Let me think.
    
    # For a 3-tensor of size N x M x P, the rank r must satisfy:
    # r >= N, r >= M, r >= P? No, that's not right either.
    
    # Actually, for the matrix multiplication tensor, a known lower bound is:
    # r >= n*m*p / (something)... I don't remember the exact bound.
    
    # Let's just try ranks from a reasonable lower bound up to 84.
    
    # For 3x4x7, let's try ranks from 30 up to 84.
    
    # Actually, let's be more systematic. We'll try several ranks and see if we can find a valid decomposition.
    
    # For each rank r, we'll:
    # 1. Initialize U, V, W with random values
    # 2. Use alternating least squares to minimize ||T - sum_t U[:,t] V[:,t] W[:,t]||_F
    # 3. If the residual is small enough, attempt rational reconstruction
    
    # Let's implement this.
    
    # First, build the tensor T as a 3D array.
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now try to find CP decompositions of various ranks.
    
    # We'll try ranks in a specific order. Let's start with some promising values.
    # Based on known results, the rank of 3x4x7 matrix multiplication is known to be at most 39
    # (or some value less than 84). Let's try a range.
    
    # Actually, let's try a more efficient approach: use the Kruskal algorithm.
    
    # For now, let's implement a simple ALS-based CP decomposition and try various ranks.
    
    best_result = None
    best_rank = N * M * P  # 84, the schoolbook rank
    
    # Try ranks from a lower bound up to best_rank - 1
    # Lower bound: Let's use 30 as a starting point (this is a guess)
    # Actually, a valid lower bound for the rank of the n x m x p matrix multiplication tensor is:
    # r >= max(n, m, p) * min(n, m, p)? No, I don't think that's right.
    
    # Let's just try a range of ranks.
    ranks_to_try = list(range(30, 85))  # Try ranks 30 to 84
    
    # For efficiency, let's try ranks in a specific order: start with lower ranks and go up.
    # But we also want to be efficient, so let's limit the number of trials per rank.
    
    np.random.seed(42)
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        found = False
        for trial in range(5):  # 5 trials per rank
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with random values
            U = np.random.randn(N, r)
            V = np.random.randn(M, r)
            W = np.random.randn(P, r)
            
            # Normalize to avoid numerical issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Alternating least squares
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: For each a, U[a, :] should satisfy:
                # sum_t U[a,t] * (V[:,t] * W[:,t])[b,c] = T[a,b,c] for all b,c
                # This is a linear system: U[a, :] * (V * W) = T[a, :, :]
                # where (V * W) is the outer product of V and W flattened
                
                # More precisely, for fixed a:
                # sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
                # Let M_a be the matrix where M_a[(b,c), t] = V[b,t] * W[c,t]
                # Then U[a, :] is the least squares solution to M_a * U[a, :] = T[a, :, :].flatten()
                
                # But this is expensive. Let's use a more efficient approach.
                
                # Actually, for ALS, we update one factor at a time:
                
                # Update U:
                # For each t, U[:, t] should be such that:
                # sum_t U[:, t] * V[:, t] * W[:, t] approximates T
                # This is not straightforward for ALS.
                
                # Let me use a different approach: unfold the tensor.
                
                # Actually, let's use a simpler approach: gradient descent with line search.
                # Or better yet, let's use the standard ALS formulation for CP decomposition.
                
                # For CP decomposition: T ≈ sum_t u_t v_t w_t
                # where u_t is N x 1, v_t is M x 1, w_t is P x 1.
                
                # ALS update for u_t (holding v_t, w_t fixed):
                # u_t = argmin_u ||T - sum_s u_s v_s w_s||_F^2
                # This gives: u_t = (sum_{b,c} T[a,b,c] v_t[b] w_t[c]) / (sum_{b,c} v_t[b]^2 w_t[c]^2)
                # Wait, that's not right either.
                
                # Let me think about this more carefully.
                
                # The objective is: min_{U,V,W} ||T - sum_t U[:,t] V[:,t] W[:,t]||_F^2
                # where U[:,t] is a vector of length N, etc.
                
                # For fixed V, W, the optimal U[:,t] is:
                # U[:,t] = argmin_u ||T - sum_s u_s V[:,s] W[:,s]||_F^2
                # Taking derivative with respect to u_t:
                # -2 sum_{a,b,c} T[a,b,c] V[a,t] W[b,t] ... 
                
                # This is getting complicated. Let me use a different approach.
                
                # I'll use the standard ALS for CP decomposition:
                # For each mode, unfold the tensor and solve a least squares problem.
                
                # Unfold T along mode 1: T_(1) is an N x (M*P) matrix where
                # T_(1)[a, b*P+c] = T[a, b, c]
                
                # The CP decomposition T ≈ sum_t U[:,t] V[:,t] W[:,t] can be written as:
                # T_(1) ≈ U * (V ⊙ W)
                # where V ⊙ W is the Khatri-Rao product of V and W, which is an (M*P) x r matrix.
                
                # So for fixed V, W: U = T_(1) (V ⊙ W)^+ (pseudo-inverse)
                
                # Similarly for V and W.
                
                # Let's implement this.
                
                # Unfold T along mode 1
                T_1 = T.reshape(N, M * P)
                
                # Compute Khatri-Rao product of V and W
                # V is M x r, W is P x r
                # V ⊙ W is (M*P) x r where (V ⊙ W)[b*P+c, t] = V[b,t] * W[c,t]
                VW = (V[:, :, None] * W[None, :, :]).reshape(M * P, r)
                
                # Solve for U: U = T_1 @ pinv(VW)
                try:
                    U_new = T_1 @ np.linalg.pinv(VW)
                except:
                    break
                
                # Update U
                U = U_new
                
                # Unfold T along mode 2
                T_2 = T.transpose(1, 0, 2).reshape(M, N * P)
                
                # Compute Khatri-Rao product of U and W
                # U is N x r, W is P x r
                # U ⊙ W is (N*P) x r where (U ⊙ W)[a*P+c, t] = U[a,t] * W[c,t]
                UW = (U[:, :, None] * W[None, :, :]).reshape(N * P, r)
                
                try:
                    V_new = T_2 @ np.linalg.pinv(UW)
                except:
                    break
                
                V = V_new
                
                # Unfold T along mode 3
                T_3 = T.transpose(2, 0, 1).reshape(P, N * M)
                
                # Compute Khatri-Rao product of U and V
                # U is N x r, V is M x r
                # U ⊙ V is (N*M) x r where (U ⊙ V)[a*M+b, t] = U[a,t] * V[b,t]
                UV = (U[:, :, None] * V[None, :, :]).reshape(N * M, r)
                
                try:
                    W_new = T_3 @ np.linalg.pinv(UV)
                except:
                    break
                
                W = W_new
                
                # Check convergence
                residual = np.linalg.norm(T - np.einsum('at,bt,ct->abc', U, V, W))
                if residual < 1e-10:
                    converged = True
                    break
            
            if not converged:
                continue
            
            # Check if the decomposition is exact (within tolerance)
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            if np.max(np.abs(T - T_recon)) < 1e-8:
                # Attempt rational reconstruction
                # Convert U, V, W to fractions
                U_frac = np.zeros((N, r), dtype=object)
                V_frac = np.zeros((M, r), dtype=object)
                W_frac = np.zeros((P, r), dtype=object)
                
                success = True
                for a in range(N):
                    for t in range(r):
                        U_frac[a, t] = Fraction(U[a, t]).limit_denominator(1000000)
                for b in range(M):
                    for t in range(r):
                        V_frac[b, t] = Fraction(V[b, t]).limit_denominator(1000000)
                for c in range(P):
                    for t in range(r):
                        W_frac[c, t] = Fraction(W[c, t]).limit_denominator(1000000)
                
                # Verify the rational decomposition is exact
                T_rational = np.zeros((N, M, P), dtype=object)
                for t in range(r):
                    for a in range(N):
                        for b in range(M):
                            for c in range(P):
                                T_rational[a, b, c] += U_frac[a, t] * V_frac[b, t] * W_frac[c, t]
                
                exact = True
                for a in range(N):
                    for b in range(M):
                        for c in range(P):
                            if T_rational[a, b, c] != T[a, b, c]:
                                exact = False
                                break
                    if not exact:
                        break
                
                if exact:
                    # Found a valid decomposition!
                    # Convert to lists for output
                    U_list = [[float(U_frac[a, t]) for t in range(r)] for a in range(N)]
                    V_list = [[float(V_frac[b, t]) for t in range(r)] for b in range(M)]
                    W_list = [[float(W_frac[c, t]) for t in range(r)] for c in range(P)]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we didn't find a valid decomposition for this rank, continue to next rank
    
    # If we didn't find a better decomposition, return the schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
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
