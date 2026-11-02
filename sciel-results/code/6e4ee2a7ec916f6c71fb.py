import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Construct tensor T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We need T[i, j, k] = 1 if i,j,k correspond to a valid multiplication term?
    # No, the tensor T is the multiplication tensor.
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]?
    # Actually, the tensor is defined such that C = A B.
    # The entries of the tensor are 1 if (i, j, k) is a valid triple for the multiplication.
    # For standard matrix multiplication, T[i, j, k] = 1 for all i, j, k?
    # No. The tensor T is such that <T, A \otimes B \otimes C^T> = sum A[i,j] B[j,k] C[k,i].
    # The "multiplication tensor" usually has entries 1 at (i, j, k) if the product A[i,j]B[j,k] is part of C[i,k].
    # In the standard basis, T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the indices in the task are:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # The condition is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
    # This means the reconstructed tensor must be all 1s.
    
    # Compute the sum
    # This is a batched matrix multiplication or einsum.
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # We can compute this efficiently.
    # Let's just check a few random points or all points if small.
    # n=3, m=4, p=7. Total points 84.
    # Rank up to 84.
    # 84 * 84 * 84 is small.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k]) # Wait, this is not right.
                # The sum is over c.
                # val = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # We can compute this as:
                # temp = U[i*m+j] * V[j*p+k] # element-wise product
                # val = np.dot(temp, W[k*n+i])
                # But doing this in a python loop is slow for 84*84*84?
                # 84*84*84 = 592,704 operations.
                # With numpy, it's fast.
                pass
    
    # Vectorized check
    # Reshape U to (n, m, rank)
    # Reshape V to (m, p, rank)
    # Reshape W to (p, n, rank)
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # We want T[i, j, k] = sum_c U_3d[i, j, c] V_3d[j, k, c] W_3d[k, i, c]
    # This is a contraction over c.
    # We can use np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # Target is all 1s
    target = np.ones((n, m, p))
    
    return np.allclose(T_recon, target, atol=1e-9)

def construct_search():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # Try ranks from 70 down to 84? No, we want lower rank.
    # Start with 70, if fail, try 75, 80, 84.
    # Actually, just try 70. If it works, great.
    
    for rank in [70, 75, 80, 84]:
        if time.time() > deadline:
            break
        attempts = 0
        while time.time() < deadline - 10:
            attempts += 1
            if attempts > 1000: # Limit attempts per rank
                break
            
            # Random sparse initialization
            # Density ~ 0.2
            U = np.random.choice([-1, 0, 1], size=(n * m, rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, rank))
            
            # Check
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # If not, we could try to optimize, but let's just try more randoms.
            # Or, use a heuristic: start with schoolbook and remove columns?
            
    # Fallback to schoolbook
    return construct()

# EVOLVE-BLOCK-START
def construct():
    # Try to find a better rank
    try:
        result = construct_search()
        return result
    except Exception as e:
        return construct()
# EVOLVE-BLOCK-END
