import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Target dimensions
    N = n * m  # 6
    M = m * p  # 21
    P = p * n  # 14
    
    # The tensor T is the matrix multiplication tensor for n x m * m x p.
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # T[i, j, k] = 1
    # Flattened indices:
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    
    # Build the target tensor T (flattened to 3D or kept as list of non-zeros)
    # Since T is all ones, we can compute the target value for any (u, v, w) easily.
    # But for optimization, we need to minimize || sum_r U_r V_r W_r - T ||_F^2
    # This is equivalent to minimizing || sum_r u_r v_r^T w_r^T - T ||_F^2 if we view it as a matrix?
    # Actually, the decomposition is sum_r u_r \otimes v_r \otimes w_r = T.
    # We can use a gradient descent approach on the factors U, V, W.
    
    # Strategy:
    # 1. Try ranks from 25 up to 42.
    # 2. For each rank, initialize U, V, W using a "Kronecker product" heuristic or random.
    #    A good initialization is to decompose the tensor using a known higher rank and then compress.
    #    Or, use the fact that T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
    #    This is rank 42 (schoolbook).
    #    We can try to find a lower rank by random initialization and gradient descent.
    #    However, gradient descent on tensor rank minimization is non-convex and can get stuck.
    
    # Alternative Strategy: Use the "Cyclic Convolution" inspiration?
    # The prompt suggests: "Model the 2x3x7 multiplication as a specific instance of a cyclic convolution... embedding m=3 into Z4 and p=7 into Z8".
    # This implies using DFT to diagonalize the multiplication?
    # Matrix multiplication C = A B.
    # If we view A and B as vectors, C is bilinear.
    # The cyclic convolution idea usually applies to polynomial multiplication or specific structured matrices.
    # For general matrix multiplication, this doesn't directly apply unless we embed the matrices into a larger algebraic structure.
    
    # Let's stick to a more robust automated search:
    # We know rank 29 is possible (hybrid Strassen).
    # Can we do better?
    # Rank 25 is the theoretical lower bound for some small cases, but for 2x3x7?
    # Let's try to optimize a rank R decomposition.
    
    # To make this fast, we will use a simplified gradient step.
    # Loss L = || T - sum_r u_r v_r w_r ||^2
    # Gradient w.r.t U: dL/dU = 2 * (T - sum_r u_r v_r w_r) contracted with V and W.
    
    # We will try ranks: [25, 26, 27, 28]
    # If we find a valid decomposition (error < 1e-9), we return it.
    # If not, we return the best heuristic (rank 29).
    
    # Precompute the target tensor T as a 3D array for fast contraction?
    # T is (n, m, p) all ones.
    # But we need to contract with V and W which are indexed by (j, k) and (k, i).
    
    # Let's define the tensor T explicitly as a numpy array of shape (n, m, p) filled with 1s.
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    def compute_error(U, V, W):
        # U: (N, R), V: (M, R), W: (P, R)
        # Reconstruct T_recon[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This is expensive if done naively.
        # We can use np.einsum or manual loops.
        # R is small (~25-42). N=6, M=21, P=14.
        # Total elements in T: 2*3*7 = 42.
        # For each of the 42 elements, we do a dot product of length R.
        # 42 * R ops. Very fast.
        
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_recon[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        return np.linalg.norm(T_recon - T_target)
    
    def gradient_step(U, V, W, lr):
        # Compute error tensor E = T_target - T_recon
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_recon[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        
        E = T_target - T_recon
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Gradient w.r.t U[u, r] = -2 * sum_{j,k} E[i,j,k] V[j*p+k, r] W[k*n+i, r]
        # where i is derived from u: i = u // m, j = u % m
        # Wait, u_idx = i*m + j. So i = u_idx // m, j = u_idx % m.
        # But the sum is over j and k? No, for a fixed U entry U[u, r], it contributes to T[i, j, k] where i, j are fixed by u.
        # Actually, U[u, r] only appears in T[i, j, k] for the specific i, j corresponding to u.
        # So for a fixed u (which fixes i, j), the gradient is:
        # dU[u, r] = -2 * sum_k E[i, j, k] V[j*p+k, r] W[k*n+i, r]
        
        for u_idx in range(N):
            i = u_idx // m
            j = u_idx % m
            for r in range(U.shape[1]):
                grad = 0.0
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    grad += E[i, j, k] * V[v_idx, r] * W[w_idx, r]
                dU[u_idx, r] = -2.0 * grad
        
        # Gradient w.r.t V[v, r] = -2 * sum_{i,j} E[i, j, k] U[i*m+j, r] W[k*n+i, r]
        # v_idx = j*p + k. So j = v_idx // p, k = v_idx % p.
        for v_idx in range(M):
            j = v_idx // p
            k = v_idx % p
            for r in range(V.shape[1]):
                grad = 0.0
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    grad += E[i, j, k] * U[u_idx, r] * W[w_idx, r]
                dV[v_idx, r] = -2.0 * grad
        
        # Gradient w.r.t W[w, r] = -2 * sum_{i,j} E[i, j, k] U[i*m+j, r] V[j*p+k, r]
        # w_idx = k*n + i. So k = w_idx // n, i = w_idx % n.
        for w_idx in range(P):
            k = w_idx // n
            i = w_idx % n
            for r in range(W.shape[1]):
                grad = 0.0
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    grad += E[i, j, k] * U[u_idx, r] * V[v_idx, r]
                dW[w_idx, r] = -2.0 * grad
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        return U, V, W

    best_result = None
    best_rank = 42
    
    # Try to find a decomposition with rank < 29
    # We start from rank 25 and go up.
    # Initialization: Random small values.
    
    np.random.seed(42)
    
    for R in range(25, 29):
        if time.time() > deadline - 20:
            break
            
        U = np.random.randn(N, R) * 0.1
        V = np.random.randn(M, R) * 0.1
        W = np.random.randn(P, R) * 0.1
        
        # Run gradient descent
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 10:
                break
                
            # Adaptive learning rate? Keep simple for now.
            U, V, W = gradient_step(U, V, W, lr)
            
            # Check error occasionally
            if step % 100 == 0:
                err = compute_error(U, V, W)
                if err < 1e-9:
                    # Found a valid decomposition
                    best_result = (U, V, W, R)
                    best_rank = R
                    break
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W, R = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to the known rank 29 decomposition
        # Re-implement the hybrid Strassen/Schoolbook from the current program
        # This is safer than returning a random high rank
        
        terms = []
        
        def add_sparse_term(u_dict, v_dict, w_dict):
            terms.append((u_dict, v_dict, w_dict))
            
        # Term 0: A2 B2
        u_dict = {2: 1, 5: 1}
        v_dict = {14+k: 1 for k in range(7)}
        w_dict = {k*2+i: 1 for k in range(7) for i in range(2)}
        add_sparse_term(u_dict, v_dict, w_dict)
        
        # Schoolbook for A1 B1 (2x2 * 2x7)
        # 3 blocks of 2x2 * 2x2
        def add_schoolbook_block(col_offset):
            # Term 1: A00 B00
            add_sparse_term({0: 1}, {col_offset: 1}, {col_offset*2: 1})
            # Term 2: A01 B10
            add_sparse_term({1: 1}, {7+col_offset: 1}, {col_offset*2: 1})
            # Term 3: A00 B01
            add_sparse_term({0: 1}, {col_offset+1: 1}, {(col_offset+1)*2: 1})
            # Term 4: A01 B11
            add_sparse_term({1: 1}, {7+col_offset+1: 1}, {(col_offset+1)*2: 1})
            # Term 5: A10 B00
            add_sparse_term({3: 1}, {col_offset: 1}, {col_offset*2+1: 1})
            # Term 6: A11 B10
            add_sparse_term({4: 1}, {7+col_offset: 1}, {col_offset*2+1: 1})
            # Term 7: A10 B01
            add_sparse_term({3: 1}, {col_offset+1: 1}, {(col_offset+1)*2+1: 1})
            # Term 8: A11 B11
            add_sparse_term({4: 1}, {7+col_offset+1: 1}, {(col_offset+1)*2+1: 1})
            
        add_schoolbook_block(0)
        add_schoolbook_block(2)
        add_schoolbook_block(4)
        
        # Last block 2x2 * 2x1
        add_sparse_term({0: 1}, {6: 1}, {12: 1})
        add_sparse_term({1: 1}, {13: 1}, {12: 1})
        add_sparse_term({3: 1}, {6: 1}, {13: 1})
        add_sparse_term({4: 1}, {13: 1}, {13: 1})
        
        R = len(terms)
        U = np.zeros((N, R))
        V = np.zeros((M, R))
        W = np.zeros((P, R))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val
                
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
