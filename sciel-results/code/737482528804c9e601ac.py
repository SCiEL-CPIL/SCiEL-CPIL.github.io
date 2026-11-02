"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Rank estimate: 
    # Schoolbook is n*m*p = 56.
    # Strassen-based estimate: 2 * (3 * R(2,2,2) + 1 * R(2,2,1))
    # R(2,2,2)=7, R(2,2,1)=2 (actually 2 is min, but standard is 2? No, 2x2x1 is rank 2? 
    # Wait, 2x2x1 tensor is just identity? No, it's A(2x2)B(2x1)=C(2x1).
    # T[i,j,0] = delta_{i,j}? No.
    # A is 2x2, B is 2x1. C[i] = sum_j A[i,j] B[j,0].
    # T[i,j,0] = 1 for all i,j? No.
    # The tensor T for multiplication is defined such that C = A B.
    # C[k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1.
    # For 2x2x1: i in 0,1; j in 0,1; k in 0.
    # T[i,j,0] = 1 for all i,j.
    # This is the all-ones tensor of shape (2,2,1).
    # Rank of all-ones (2,2,1) is 2?
    # T = u v w^T? 
    # If rank 1: T[i,j,0] = u_i v_j w_0.
    # u_i v_j must be constant 1. So u=[1,1], v=[1,1]. Rank 1?
    # Let's check. 
    # A=[[1,1],[1,1]], B=[[1],[1]]. C=[[2],[2]].
    # T[i,j,0]=1.
    # Sum_d U[i,d] V[j,d] W[0,d] = 1.
    # If rank 1: U[0]V[0]W[0]=1, U[1]V[0]W[0]=1, U[0]V[1]W[0]=1, U[1]V[1]W[0]=1.
    # U[0]=U[1], V[0]=V[1].
    # So yes, rank 1 is possible for 2x2x1?
    # Wait, standard matrix mult 2x2x1 (C is 2x1 vector) requires 2 multiplications?
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # We need to compute 2 outputs.
    # The tensor T has entries T[i,j,k]. Here k is always 0.
    # T[i,j,0] = 1.
    # This tensor is rank 1? 
    # T = [1,1]^T \otimes [1,1]^T \otimes [1]^T.
    # Yes, rank 1.
    # So R(2,2,1) = 1?
    # Let's re-read the knowledge: "R(2,2,1)=4" was in one finding, "R(2,2,1)=2" in another.
    # Actually, let's look at the tensor definition again.
    # The tensor for matrix multiplication $C = AB$ is $T_{i,j,k} = 1$.
    # For $A$ (2x2), $B$ (2x1), $C$ (2x1).
    # Indices: i (row of A/C), j (col of A/row of B), k (col of B/C).
    # k is only 0.
    # T[i,j,0] = 1 for all i,j in {0,1}.
    # This is indeed the all-ones tensor, which has rank 1.
    # So 2x2x1 rank is 1.
    # Then total rank = 2 * (3 * 7 + 1 * 1) = 2 * 22 = 44.
    
    # Let's try rank 44. If it fails, try 46, 48, 50, 52, 54, 56.
    
    ranks_to_try = [44, 46, 48, 50, 52, 54, 56]
    
    best_result = None
    best_error = float('inf')
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Initialize U, V, W
        # Using small random initialization
        np.random.seed(42 + rank)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
                
            U_flat = x[:nm*rank]
            V_flat = x[nm*rank:2*nm*rank]
            W_flat = x[2*nm*rank:]
            
            U_curr = U_flat.reshape(nm, rank)
            V_curr = V_flat.reshape(mp, rank)
            W_curr = W_flat.reshape(pn, rank)
            
            # Compute reconstruction T_rec = sum_d U[:,d] V[:,d] W[:,d]^T
            # Efficient way: einsum
            # T_rec[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
            T_rec = np.einsum('ad,bd,cd->abc', U_curr, V_curr, W_curr)
            
            # Error
            err = np.linalg.norm(T - T_rec, 'fro')
            return err
        
        # Run optimization
        try:
            # BFGS is good for smooth problems
            # We don't need gradients for a rough search, but finite difference is slow.
            # Let's try a few steps of BFGS or just L-BFGS-B
            # Since dimension is 8*44 + 28*44 + 14*44 = 352 + 1232 + 616 = 2200 variables.
            # This is large.
            
            # Alternative: Use a simpler heuristic or smaller rank first.
            # Actually, let's use `minimize` with method 'BFGS' and limited iterations.
            
            # To speed up, we can use a structured initialization based on Strassen for the first few.
            # But random is safer.
            
            # Let's run a quick check with a small number of iterations
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-5})
            
            if res.fun < best_error:
                best_error = res.fun
                if res.fun < 1e-5:
                    # Success
                    U_final = res.x[:nm*rank].reshape(nm, rank)
                    V_final = res.x[nm*rank:2*nm*rank].reshape(mp, rank)
                    W_final = res.x[2*nm*rank:].reshape(pn, rank)
                    
                    # Verify
                    T_rec = np.einsum('ad,bd,cd->abc', U_final, V_final, W_final)
                    if np.allclose(T, T_rec, atol=1e-5):
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        
        except Exception:
            continue
            
    # If optimization didn't find a perfect solution, return the best attempt or fallback to schoolbook
    # Schoolbook rank 56 is guaranteed.
    if best_result is None:
        # Construct Schoolbook
        rank = nm * mp * pn # No, schoolbook is n*m*p = 56
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                    idx += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback if no perfect solution found but we have a best error
    # Just return schoolbook to be safe
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
