"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal delta[i,k] (since we're multiplying A[i,j]*B[j,k] and summing over j)
    # Wait, the tensor for matrix multiplication is T[i,j,k] where the multiplication is
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 if this term contributes, which it always does.
    # Actually the decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    # No, let me think again.
    # The standard tensor decomposition for matmul: 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i,j] B[j,k]
    # For this to hold, we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta[i,k]... 
    # No wait. Let me re-read the contract.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # The decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
    # With tensor T where T[i,j,k] represents the coefficient.
    # Actually for matmul, the "tensor" is just the identity in the right sense.
    # Let me just verify: for each (i,j,k), the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c
    # should give 1 if the term A[i,j]*B[j,k] appears in C[i,k], which is always true.
    # So T[i,j,k] = 1 for all i,j,k? No that's not right either.
    # 
    # Actually, I think the convention is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition gives: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Hmm, that doesn't look standard. Let me just check with schoolbook.
    # In schoolbook, for each (i,j,k), there's a column with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for the specific (i,j,k) where all three are 1, else 0.
    # Wait no, each column corresponds to one (i,j,k) triple, so for a given (i,j,k), exactly one column has all three nonzero.
    # So the product summed over c gives 1 for each (i,j,k).
    # That means T[i,j,k] = 1 for all i,j,k, which is the all-ones tensor.
    # But that's not the matmul tensor...
    # 
    # I think I'm overcomplicating this. The schoolbook is known to be correct, so let me just verify
    # that my candidate has the same "tensor" as schoolbook.
    U_school = np.array(_schoolbook()["U"], dtype=float)
    V_school = np.array(_schoolbook()["V"], dtype=float)
    W_school = np.array(_schoolbook()["W"], dtype=float)
    
    def compute_tensor(U_arr, V_arr, W_arr):
        r = U_arr.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U_arr[:, c]
            v = V_arr[:, c]
            w = W_arr[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        return T
    
    T_target = compute_tensor(U_school, V_school, W_school)
    T_candidate = compute_tensor(U, V, W)
    return np.allclose(T_target, T_candidate, atol=1e-9)

def _is_halfint(x):
    """Check if x is an integer or half-integer."""
    x = float(x)
    return abs(x * 2 - round(x * 2)) < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    best = _schoolbook()
    best_rank = best["U"][0].__len__() if best["U"] else 0
    n, m, p = DIMENSIONS
    
    # Try structured decompositions with rank < n*m*p
    # Strategy: For matmul, we can sometimes reduce rank by exploiting structure.
    # For 3x5 * 5x5, the schoolbook rank is 75.
    # Let's try to find a decomposition with lower rank.
    
    # Approach: Use the fact that the matmul tensor has a specific structure.
    # We can try to find U, V, W with fewer columns.
    
    # Try ranks from 75 down
    for rank in range(74, 40, -1):
        if time.time() > deadline:
            break
        
        # Try to construct a decomposition of this rank
        # Use a greedy approach: start with schoolbook and try to merge columns
        
        # Alternative: try to find a low-rank approximation that is exact
        # This is hard, so let's try a different approach.
        
        # For now, just return schoolbook if we can't find better
        pass
    
    return best
# EVOLVE-BLOCK-END
