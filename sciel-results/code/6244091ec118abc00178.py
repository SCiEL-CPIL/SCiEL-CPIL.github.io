"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # The tensor T[i,j,k,l] = delta[i,l] * delta[j,k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Our decomposition: T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention. W[k*n+i, r]
    # So for output C[i, k] = sum_j A[i,j] * B[j,k]
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # Actually the tensor is T[i,j,k] = delta[i,k]... no.
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form: sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # Tensor T[i,j,k] such that sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] = sum_{i,k} (A*B)[i,k]
    # So T[i,j,k] = delta[i,k]... no, that's not right either.
    # Actually: sum_{i,j,k} A[i,j] B[j,k] * (something) should equal sum_{i,k} (A*B)[i,k] * 1
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So sum_{i,k} (A*B)[i,k] = sum_{i,k} sum_j A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that can't be right for a tensor decomposition.
    
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is the column of B (and C), i is the row of A (and C)
    # U is indexed by (i, j) - row i, col j of A
    # V is indexed by (j, k) - row j, col k of B
    
    # The decomposition: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_k B[j,k] V[j*p+k, r]) ... no
    
    # Standard tensor decomposition for matmul:
    # (A*B)[i,k] = sum_r U[i*m + j, r]... 
    
    # Actually the standard form is:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has indices (i, j, k) where T[i,j,k] = 1 if we're computing (A*B)[i,k]
    # Wait, I think the tensor is: T[i, j, k] such that the bilinear form is
    # sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] = sum_{i,k} (A*B)[i,k]
    # This gives T[i,j,k] = 1 for all valid indices? No...
    
    # Let me think again. The matrix multiplication tensor M has entries:
    # M[i, j, k] = 1 if the (i,k)-th entry of A*B involves A[i,j] * B[j,k]
    # So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that's just the all-ones tensor.
    
    # I think the correct interpretation is:
    # We want: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] is defined so that sum_{j} T[i,j,k] A[i,j] B[j,k] ... no.
    
    # Let me look at the factor shapes:
    # U: (n*m, rank) - indexed by (i,j)
    # V: (m*p, rank) - indexed by (j,k)  
    # W: (p*n, rank) - indexed by (k,i)
    
    # The reconstruction is: (A*B)[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * ... 
    # No, that's not standard.
    
    # I believe the correct formula is:
    # (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j?
    # No, that doesn't make sense dimensionally.
    
    # Standard Strassen-style:
    # (A*B)[i,k] = sum_r (sum_j A[i,j] * U_{(i,j),r}) * (sum_k B[j,k] * V_{(j,k),r}) * W_{(k,i),r}
    # But this sums over j and k inside, which is wrong.
    
    # I think the correct interpretation is:
    # Define intermediate products: P_r = (A * U_r) * (V_r * B) where U_r and V_r are 
    # coefficient vectors, and then (A*B) = sum_r W_r * P_r
    # Where P_r[i,k] = (sum_j A[i,j] U_{(i,j),r}) * (sum_j B[j,k] V_{(j,k),r})
    # And then (A*B)[i,k] = sum_r W_{(k,i),r} * P_r[i,k]
    
    # Let me verify with the Strassen example in the code:
    # P1 = (A00 + A11)(B00 + B11)
    # U: U[0,0]=1, U[1,1]=1 (for the first factor)
    # V: V[0,0]=1, V[1,1]=1 (for the second factor)
    # W: W[0,0]=1, W[1,1]=1
    # So P1[0,0] = (A00*1 + A11*1... wait, that's not right.
    
    # I think U_{(i,j),r} is the coefficient for A[i,j] in the r-th intermediate product's first factor.
    # So first factor of P_r at position (i, ?) = sum_j A[i,j] * U_{(i,j),r}
    # But this gives a vector, not a matrix.
    
    # Let me reconsider. In Strassen's algorithm:
    # M1 = (A00+A11)(B00+B11)
    # Here, the first factor is (A00+A11) which is a 2x2 matrix, and B00+B11 is also 2x2.
    # So U and V must define how to form these submatrices.
    
    # I think the correct interpretation is:
    # For each r, we form:
    #   A_r = sum_{i,j} U_{(i,j),r} * A_{sub(i,j)}  where A_{sub} are blocks
    #   B_r = sum_{j,k} V_{(j,k),r} * B_{sub(j,k)}
    #   P_r = A_r * B_r
    #   (A*B)_{sub(i,k)} += W_{(k,i),r} * P_r_{sub(i,k)}
    
    # For 2x2x2 Strassen, A and B are 2x2 blocks, so A_{sub(i,j)} = A[i,j] (scalar)
    # A_r = sum_{i,j} U_{(i,j),r} * A[i,j] -- this is a scalar!
    # B_r = sum_{j,k} V_{(j,k),r} * B[j,k] -- scalar
    # P_r = A_r * B_r -- scalar
    # (A*B)[i,k] += W_{(k,i),r} * P_r
    
    # Let's verify: P1 = (A00+A11)(B00+B11)
    # A_1 = A00 + A11, B_1 = B00 + B11, P_1 = (A00+A11)(B00+B11)
    # (A*B)[0,0] += W[0,0,1] * P_1 = 1 * P_1
    # (A*B)[1,1] += W[1,1,1] * P_1 = 1 * P_1
    # This matches!
    
    # So the formula is:
    # (A*B)[i,k] = sum_r W_{(k,i),r} * (sum_{i',j} U_{(i',j),r} * A[i',j]) * (sum_{j',k'} V_{(j',k'),r} * B[j',k'])
    
    # Wait, but U is indexed by (i,j) globally, not by block. For 2x2x2:
    # U has shape (2*2, 7) = (4, 7)
    # U[0,0] corresponds to (i=0,j=0), U[0,1] to (i=0,j=1), etc.
    
    # Let me re-verify P1:
    # U[0,0]=1, U[1,1]=1 (indices: i=0,j=0 and i=1,j=1)
    # A_1 = A[0,0]*1 + A[1,1]*1 = A00 + A11 ✓
    # V[0,0]=1, V[1,1]=1
    # B_1 = B[0,0]*1 + B[1,1]*1 = B00 + B11 ✓
    # W[0,0]=1, W[1,1]=1
    # (A*B)[0,0] += 1 * P_1 ✓
    # (A*B)[1,1] += 1 * P_1 ✓
    
    # Great, so the formula is confirmed. Now let me implement verification.
    
    # For general n,m,p:
    # (A*B)[i,k] = sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    
    # To verify, we can check this identity on random matrices.
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    AB = A @ B
    
    # Compute via decomposition
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # A_r = U^T @ A.flatten() for each r
    # A_r[i'] = sum_j A[i',j] * U[i'*m+j, r]
    # This is: for each r, A_r = (U[:, r].reshape(n,m) * A).sum(axis=1) ... no
    # A_r_scalar = sum_{i',j} U[i'*m+j, r] * A[i',j]
    # = (U[:, r].reshape(n,m) * A).sum()
    
    # B_r_scalar = sum_{j',k'} V[j'*p+k', r] * B[j',k']
    # = (V[:, r].reshape(m,p) * B).sum()
    
    # (A*B)[i,k] = sum_r W[k*n+i, r] * A_r_scalar * B_r_scalar
    
    # Let's compute:
    # A_r = (U[:, r].reshape(n,m) * A).sum() for each r
    # B_r = (V[:, r].reshape(m,p) * B).sum() for each r
    # Result[i,k] = sum_r W[k*n+i, r] * A_r * B_r
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # A_r for all r: A_flat @ U  -> shape (rank,)
    A_r = A_flat @ U  # shape (rank,)
    B_r = B_flat @ V  # shape (rank,)
    
    # Result[i,k] = sum_r W[k*n+i, r] * A_r[r] * B_r[r]
    # = (W * (A_r * B_r)).sum(axis=1) ... 
    # W has shape (p*n, rank), W[k*n+i, r]
    # Result_flat[k*n+i] = sum_r W[k*n+i, r] * A_r[r] * B_r[r]
    # = (W * (A_r * B_r)[None, :]).sum(axis=1)
    
    prod = A_r * B_r  # shape (rank,)
    Result_flat = (W * prod[None, :]).sum(axis=1)  # shape (p*n,)
    Result = Result_flat.reshape(p, n).T  # shape (n, p)
    
    diff = np.abs(Result - AB).max()
    return diff < 1e-10

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (6, 17)
    # V: (m*p, rank) -> (9, 17)
    # W: (p*n, rank) -> (6, 17)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def idx_U(i, j): return i * m + j
    def idx_V(j, k): return j * p + k
    def idx_W(k, i): return k * n + i
    
    r = 0
    
    # --- Strassen for 2x2x2 block (A[0:2, 0:2] * B[0:2, 0:2]) ---
    # P1 = (A00 + A11)(B00 + B11)
    U[idx_U(0,0)][r] += 1.0
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(0,0)][r] += 1.0
    V[idx_V(1,1)][r] += 1.0
    W[idx_W(0,0)][r] += 1.0  # C00
    W[idx_W(1,1)][r] += 1.0  # C11
    r += 1
    
    # P2 = (A10 + A11)B00
    U[idx_U(1,0)][r] += 1.0
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(0,0)][r] += 1.0
    W[idx_W(0,1)][r] += 1.0  # C10
    W[idx_W(1,1)][r] += -1.0 # C11
    r += 1
    
    # P3 = A00(B01 - B11)
    U[idx_U(0,0)][r] += 1.0
    V[idx_V(0,1)][r] += 1.0
    V[idx_V(1,1)][r] += -1.0
    W[idx_W(1,0)][r] += 1.0  # C01
    W[idx_W(1,1)][r] += 1.0  # C11
    r += 1
    
    # P4 = A11(B10 - B00)
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(1,0)][r] += 1.0
    V[idx_V(0,0)][r] += -1.0
    W[idx_W(0,0)][r] += 1.0  # C00
    W[idx_W(0,1)][r] += 1.0  # C10
    r += 1
    
    # P5 = (A00 + A01)B11
    U[idx_U(0,0)][r] += 1.0
    U[idx_U(0,1)][r] += 1.0
    V[idx_V(1,1)][r] += 1.0
    W[idx_W(0,0)][r] += -1.0 # C00
    W[idx_W(1,0)][r] += 1.0  # C01
    r += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[idx_U(1,0)][r] += 1.0
    U[idx_U(0,0)][r] += -1.0
    V[idx_V(0,0)][r] += 1.0
    V[idx_V(0,1)][r] += 1.0
    W[idx_W(1,1)][r] += 1.0  # C11
    r += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[idx_U(0,1)][r] += 1.0
    U[idx_U(1,1)][r] += -1.0
    V[idx_V(1,0)][r] += 1.0
    V[idx_V(1,1)][r] += 1.0
    W[idx_W(0,0)][r] += 1.0  # C00
    r += 1
    
    # --- Outer Product for A[0:2, 2] * B[2, 0:2] ---
    # Q1 = A02 B20 -> C00
    U[idx_U(0,2)][r] += 1.0
    V[idx_V(2,0)][r] += 1.0
    W[idx_W(0,0)][r] += 1.0
    r += 1
    
    # Q2 = A02 B21 -> C01
    U[idx_U(0,2)][r] += 1.0
    V[idx_V(2,1)][r] += 1.0
    W[idx_W(1,0)][r] += 1.0
    r += 1
    
    # Q3 = A12 B20 -> C10
    U[idx_U(1,2)][r] += 1.0
    V[idx_V(2,0)][r] += 1.0
    W[idx_W(0,1)][r] += 1.0
    r += 1
    
    # Q4 = A12 B21 -> C11
    U[idx_U(1,2)][r] += 1.0
    V[idx_V(2,1)][r] += 1.0
    W[idx_W(1,1)][r] += 1.0
    r += 1
    
    # --- Matrix-Vector for A[0:2, 0:3] * B[0:3, 2] ---
    # R1 = A00 B02 -> C02
    U[idx_U(0,0)][r] += 1.0
    V[idx_V(0,2)][r] += 1.0
    W[idx_W(2,0)][r] += 1.0
    r += 1
    
    # R2 = A01 B12 -> C02
    U[idx_U(0,1)][r] += 1.0
    V[idx_V(1,2)][r] += 1.0
    W[idx_W(2,0)][r] += 1.0
    r += 1
    
    # R3 = A02 B22 -> C02
    U[idx_U(0,2)][r] += 1.0
    V[idx_V(2,2)][r] += 1.0
    W[idx_W(2,0)][r] += 1.0
    r += 1
    
    # R4 = A10 B02 -> C12
    U[idx_U(1,0)][r] += 1.0
    V[idx_V(0,2)][r] += 1.0
    W[idx_W(2,1)][r] += 1.0
    r += 1
    
    # R5 = A11 B12 -> C12
    U[idx_U(1,1)][r] += 1.0
    V[idx_V(1,2)][r] += 1.0
    W[idx_W(2,1)][r] += 1.0
    r += 1
    
    # R6 = A12 B22 -> C12
    U[idx_U(1,2)][r] += 1.0
    V[idx_V(2,2)][r] += 1.0
    W[idx_W(2,1)][r] += 1.0
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
