import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    for R in [32, 36, 40, 44, 48, 52, 56, 60]:
        if R >= best_rank:
            break
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial)
            
            # Initialize
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Optimize
            def loss_and_grad(params):
                U_flat = params[:n*m*R]
                V_flat = params[n*m*R:n*m*R+m*p*R]
                W_flat = params[n*m*R+m*p*R:n*m*R+m*p*R+p*n*R]
                
                U = U_flat.reshape(n*m, R)
                V = V_flat.reshape(m*p, R)
                W = W_flat.reshape(p*n, R)
                
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            lhs = U[i*m+j] @ V[j*p+k]
                            rhs = W[k*n+i]
                            loss += np.sum((lhs - rhs)**2)
                
                # Compute gradients (simplified, using finite differences would be too slow)
                # For now, just return loss
                return loss
            
            # Use simple gradient descent with finite differences
            # This is too slow, let's use a different approach
            
            # Instead, let's try to solve the linear system for W given U and V
            # For each r, the constraint is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
            # This is nonlinear, so we need a different approach.
            
            # Let's try a different strategy: fix U and V, solve for W
            # But W appears on both sides...
            
            # Actually, the equation is:
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] = W[k*n+i,r]
            # This means W[k*n+i, r] = sum_r' U[i*m+j,r'] * V[j*p+k,r'] where r'=r
            # Wait, that's not right. Let me re-read.
            
            # The tensor equation is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
            # But this doesn't make sense dimensionally. Let me re-check.
            
            # Actually, I think the convention is:
            # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
            # And the tensor decomposition is:
            # (A * B)[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # No, that's not right either.
            
            # Let me re-read the problem statement.
            # "U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]"
            # This means for each r, the product U[i*m+j,r] * V[j*p+k,r] should equal W[k*n+i,r]
            # But that can't be right because the left side depends on i,j,k and the right side depends on k,i.
            
            # I think I'm misunderstanding the convention. Let me re-read.
            # "The output factor follows the official trace convention W[k*n+i,r]"
            # So W is indexed by (k*n+i, r)
            
            # The tensor equation for matrix multiplication is:
            # C[i, k] = sum_j A[i, j] * B[j, k]
            # In tensor form: T[i,j,k] = A[i,j] * B[j,k]
            # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
            # But the problem says U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
            
            # So the equation is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ???
            # This still doesn't make sense.
            
            # Wait, I think the decomposition is:
            # The matrix multiplication map is: (A, B) -> A*B
            # This is a bilinear map, which can be decomposed as:
            # (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[j,r]) * (sum_k B[j,k] V[r,k]) * W[r,i,k]
            # No, that's not standard.
            
            # Let me look at the schoolbook example again:
            # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
            # For a fixed (i,j,k), col = i*m*p + j*p + k
            # So U[i*m+j, col] = 1 only when the column corresponds to (i,j,k)
            # Similarly for V and W.
            
            # So the tensor equation must be:
            # sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = ?
            
            # But the problem says the score is the rank, and the decomposition should be exact.
            # I think the convention is that the product of the three matrices gives the tensor.
            
            # Actually, I think the correct interpretation is:
            # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
            # And the decomposition is T = sum_r U_r ⊗ V_r ⊗ W_r
            # Where U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n
            
            # So T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # But the tensor T is 1 for all (i,j,k) in the appropriate range, and 0 otherwise.
            # Actually, T is defined on the full space, but we only care about the entries where i,j,k are in range.
            
            # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # And for all other (i,j,k), the sum should be 0 (but since we only have entries for valid i,j,k, this might not be needed).
            
            # Wait, but the schoolbook has U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for col corresponding to (i,j,k)
            # So sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = 1 (only the col for (i,j,k) contributes)
            # And for a different (i',j',k'), the sum would be 0 if (i',j',k') != (i,j,k)
            
            # So the condition is: the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), and the decomposition should reproduce this.
            
            # But that means we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
            # And = 0 for all invalid (i,j,k) (but there are no invalid entries in the domain)
            
            # Actually, the tensor is defined on the full Cartesian product, so we need the sum to be 1 for valid (i,j,k) and 0 for invalid ones.
            # But since i, j, k are always in range in our indexing, we only need the sum to be 1 for all (i,j,k) in [0,n)×[0,m)×[0,p).
            
            # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This makes more sense!
            
            # So I need to solve:
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a system of 64 equations in 56R unknowns.
            
            # For R=32, we have 1792 unknowns and 64 equations, so it's very underdetermined.
            
            # Let me try to find a solution.
            
            pass
        
        if not found:
            continue
    
    return best_result
