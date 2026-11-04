import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    # The matrix multiplication tensor T_{ijk} = delta_{i,k} (for C[i,k] = sum_j A[i,j]B[j,k])
    # Wait, the tensor is defined by the bilinear map.
    # C_{ik} = sum_{j} A_{ij} B_{jk}
    # In tensor decomposition form:
    # C_{ik} = sum_{r} U_{i,j_r} V_{j_r,k} W_{k,i} ? No.
    # The standard tensor for matrix multiplication is T_{i,j,k} where the index i is row of A/C,
    # j is col of A/row of B, k is col of B/C.
    # The structure is: T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, the tensor is just the identity in the "middle" index?
    # No, the tensor M_{ijk} such that M(A,B)_{ik} = sum_{j} A_{ij} B_{jk} M_{ijk}?
    # No, the tensor is the "multiplication tensor" mu.
    # mu_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # Since C_{ik} = sum_j A_{ij} B_{jk}, the coefficient is 1 for all i,j,k.
    # So the tensor is all ones? No, that would mean C_{ik} = sum_{i,j,k} A_{ij} B_{jk}.
    # The indices of the tensor decomposition are:
    # U is (n*m) x R
    # V is (m*p) x R
    # W is (p*n) x R
    # The reconstruction is:
    # C_{ik} = sum_{r} (sum_{j} U_{i*m+j, r} A_{ij}) (sum_{j'} V_{j'*p+k, r} B_{j'k}) W_{k*n+i, r}
    # This is not quite right. The standard definition is:
    # C_{ik} = sum_{r} (A \cdot U_r) (B \cdot V_r) W_r
    # Where U_r is a matrix? No, U_r is a vector in R^{n*m}.
    # Let's use the flattened form.
    # vec(C) = sum_r vec(A * U_r) ... no.
    
    # Correct formulation:
    # We want to find rank R such that:
    # C_{ik} = sum_{r=0}^{R-1} (sum_{j} A_{ij} U_{i,j,r}) (sum_{j} B_{jk} V_{j,k,r}) W_{i,k,r}
    # Wait, U, V, W are usually defined as:
    # U_{i,j,r}, V_{j,k,r}, W_{i,k,r}
    # Such that C_{ik} = sum_{r} sum_{j} A_{ij} B_{jk} ... no, the sum over j is inside the factors?
    # No, the rank-1 term is:
    # (sum_j A_{ij} u_{ij,r}) (sum_j B_{jk} v_{jk,r}) w_{ik,r}
    # This is not bilinear in A and B if we sum j inside.
    # A rank-1 term in the tensor decomposition of the multiplication tensor is:
    # T_{i,j,k} = u_{i,j} v_{j,k} w_{i,k}
    # And the product is:
    # C_{ik} = sum_{r} (sum_{j} A_{ij} u_{i,j,r}) (sum_{j} B_{jk} v_{j,k,r}) w_{i,k,r}
    # This is still not standard.
    
    # Let's look at the schoolbook decomposition provided in the prompt:
    # For each (i, j, k):
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # This implies the term is:
    # (A_{ij}) (B_{jk}) (1)
    # And it contributes to C_{ik}.
    # So the general term r is:
    # (sum_{i',j'} A_{i'j'} U_{i'm+j', r}) (sum_{j',k'} B_{j'k'} V_{j'p+k', r}) W_{kn+i, r}
    # Wait, if U_{i'm+j', r} is non-zero only for specific i', j', then the first factor is just a linear combination of A's elements.
    # If U is sparse, it's just A_{ij}.
    # The reconstruction is:
    # C_{ik} = sum_{r} W_{kn+i, r} * (sum_{j} A_{ij} U_{i*m+j, r}) * (sum_{j} B_{jk} V_{j*p+k, r})
    # This matches the schoolbook if U, V are identity-like and W is 1.
    
    # So the goal is to find U, V, W of rank R such that:
    # C_{ik} = sum_{r} W_{kn+i, r} (sum_{j} A_{ij} U_{i*m+j, r}) (sum_{j} B_{jk} V_{j*p+k, r})
    # holds for all A, B.
    # This is equivalent to:
    # 1 = sum_{r} W_{kn+i, r} (sum_{j} delta_{i,i'} delta_{j,j'} U_{i'm+j', r}) (sum_{j} delta_{j,j''} delta_{k,k'} V_{j'p+k', r})
    # for all i,k,i',j',j'',k'? No.
    # The tensor T_{i,j,k} (coefficient of A_{ij}B_{jk} in C_{ik}) is 1.
    # The decomposition approximates T.
    # T_{i,j,k} = sum_{r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # Wait, the indices of W in the schoolbook are W[k*n+i, r].
    # So the condition is:
    # 1 = sum_{r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # This is a system of n*m*p equations.
    # We want to minimize R such that this system has a solution.
    # This is a rank minimization problem for a 3-way tensor T of size n*m x m*p x p*n?
    # No, T is indexed by (i,j) for U, (j,k) for V, (k,i) for W.
    # Let's reshape T into a 3D array of shape (n*m, m*p, p*n).
    # T_flat[i*m+j, j*p+k, k*n+i] = 1.
    # We want to find U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # T_flat = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # This is exactly the CP decomposition of the tensor T_flat.
    # The tensor T_flat is sparse. It has 1s at (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
    # Note that the indices are coupled.
    
    # Strategy:
    # 1. Construct the sparse tensor T_flat.
    # 2. Use a CP decomposition algorithm to find a low-rank approximation.
    # 3. Verify if the approximation is exact (within tolerance).
    # 4. If exact, check if the entries are "nice" (integers/half-integers) or if we can round them.
    #    The problem states "integer/half-integer entries".
    #    Numerical methods might produce floats. We need to check if they are close to half-integers.
    #    If not, we might need to search for an exact rational decomposition.
    #    However, for small sizes, the minimal rank is often achieved by "nice" decompositions.
    
    # Let's try to find the minimal rank R by trying R from lower bound to upper bound.
    # Lower bound: max(n*m, m*p, p*n) / something? No, rank is at least max dimension?
    # For matrix multiplication, rank >= n*m*p / max(n,m,p)^2? No.
    # Rank >= max(n*m, m*p, p*n) is not true.
    # Rank >= n+m+p-2 is a lower bound for some? No.
    # Let's just try R from 10 to 100.
    
    # To make it efficient, we can use a random initialization and gradient descent
    # to minimize the error ||T_flat - sum_r U_r V_r W_r||_F.
    # If the error drops to ~0, we have a decomposition.
    
    # Construct T_flat
    T_flat = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_flat[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We will try to find a rank-R decomposition.
    # We'll use a simple gradient descent on the factors U, V, W.
    # Objective: L(U,V,W) = 0.5 * ||T_flat - sum_r U[:,r] outer V[:,r] outer W[:,r]||_F^2
    
    best_result = None
    best_rank = n*m*p # 100
    
    # Try ranks from 20 up to 100
    # We start low to find the minimum.
    # But if R is too low, it might not converge to 0.
    # We'll iterate R from 20 to 100.
    # To save time, we'll do a few restarts for each R.
    
    for R in range(20, 101):
        if time.time() > deadline - 10: break
        
        # Try a few random initializations
        for restart in range(3):
            if time.time() > deadline - 5: break
            
            # Initialize U, V, W
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Normalize to prevent explosion
            for r in range(R):
                norm_u = np.linalg.norm(U[:,r])
                norm_v = np.linalg.norm(V[:,r])
                norm_w = np.linalg.norm(W[:,r])
                if norm_u > 0 and norm_v > 0 and norm_w > 0:
                    # Scale such that product of norms is 1
                    scale = 1.0 / (norm_u * norm_v * norm_w) ** (1.0/3.0)
                    U[:,r] *= scale
                    V[:,r] *= scale
                    W[:,r] *= scale
            
            # Gradient Descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 2: break
                
                # Compute reconstruction
                # C_approx[i,j,k_idx] = sum_r U[i,j,r] V[j,k_idx,r] W[k_idx,i,r]
                # Note: V index is j*p+k, W index is k*n+i
                # Let's use einsum for speed
                # T_approx = np.einsum('ijr,jkr,ikr->ijk', U, V, W) 
                # But indices are flattened.
                # U is (n*m, R), V is (m*p, R), W is (p*n, R)
                # We need to map (i,j) -> i*m+j, (j,k) -> j*p+k, (k,i) -> k*n+i
                # This is tricky with einsum directly on flattened arrays without reshaping.
                # Let's reshape U, V, W to 3D arrays for the contraction.
                # U_3d[i, j, r] = U[i*m+j, r]
                # V_3d[j, k, r] = V[j*p+k, r]
                # W_3d[k, i, r] = W[k*n+i, r]
                
                U_3d = U.reshape(n, m, R)
                V_3d = V.reshape(m, p, R)
                W_3d = W.reshape(p, n, R) # Note: W is indexed by k*n+i, so shape (p, n, R)
                
                # T_approx[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                # Note: W_3d is indexed by k, i.
                T_approx = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                # Error
                T_error = T_approx - T_flat
                err = np.linalg.norm(T_error)
                
                if err < 1e-6:
                    # Found a solution!
                    # Check if entries are close to half-integers
                    U_flat = U.flatten()
                    V_flat = V.flatten()
                    W_flat = W.flatten()
                    
                    # Check if we can round to half-integers
                    U_rounded = np.round(U_flat * 2) / 2
                    V_rounded = np.round(V_flat * 2) / 2
                    W_rounded = np.round(W_flat * 2) / 2
                    
                    # Verify rounded solution
                    U_r_3d = U_rounded.reshape(n, m, R)
                    V_r_3d = V_rounded.reshape(m, p, R)
                    W_r_3d = W_rounded.reshape(p, n, R)
                    T_approx_r = np.einsum('ijr,jkr,kir->ijk', U_r_3d, V_r_3d, W_r_3d)
                    err_r = np.linalg.norm(T_approx_r - T_flat)
                    
                    if err_r < 1e-6:
                        # Valid half-integer solution
                        if R < best_rank:
                            best_rank = R
                            best_result = {
                                "U": U_rounded.reshape(n*m, R).tolist(),
                                "V": V_rounded.reshape(m*p, R).tolist(),
                                "W": W_rounded.reshape(p*n, R).tolist()
                            }
                        break # Found best for this R, break restarts
                    else:
                        # Not half-integer, but maybe we can use it?
                        # The problem requires integer/half-integer entries.
                        # If we can't round, this solution is invalid for the score.
                        # But maybe we can try to snap to integers?
                        pass
                    
                    break # Break gradient descent steps
                
                # Compute gradients
                # dL/dU_3d[i,j,r] = 2 * sum_{j',k'} T_error[i,j',k'] V_3d[j',k',r] W_3d[k',i,r] ?
                # L = 0.5 * sum_{i,j,k} (T_approx[i,j,k] - T_flat[i,j,k])^2
                # dL/dU[i,j,r] = sum_{j',k'} (T_approx[i,j',k'] - T_flat[i,j',k']) * V[j',k',r] * W[k',i,r]
                # Note: In T_approx[i,j',k'], U is at [i,j',r].
                # So for a fixed i, j, r, we sum over j', k' where U[i,j',r] is involved?
                # No, U[i,j,r] is only involved in T_approx[i,j,k] for all k.
                # T_approx[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
                # So dL/dU[i,j,r] = sum_k (T_approx[i,j,k] - T_flat[i,j,k]) * V[j,k,r] * W[k,i,r]
                
                dU = np.einsum('ijk,jkr,kir->ijr', T_error, V_3d, W_3d)
                dV = np.einsum('ijk,ijr,kir->jkr', T_error, U_3d, W_3d)
                dW = np.einsum('ijk,ijr,jkr->kir', T_error, U_3d, V_3d)
                
                # Update
                U -= lr * dU.reshape(n*m, R)
                V -= lr * dV.reshape(m*p, R)
                W -= lr * dW.reshape(p*n, R)
                
                # Renormalize to prevent drift
                for r in range(R):
                    norm_u = np.linalg.norm(U[:,r])
                    norm_v = np.linalg.norm(V[:,r])
                    norm_w = np.linalg.norm(W[:,r])
                    if norm_u > 0 and norm_v > 0 and norm_w > 0:
                        scale = 1.0 / (norm_u * norm_v * norm_w) ** (1.0/3.0)
                        U[:,r] *= scale
                        V[:,r] *= scale
                        W[:,r] *= scale
    
    if best_result is None:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        best_result = {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
        
    return best_result
# EVOLVE-BLOCK-END
