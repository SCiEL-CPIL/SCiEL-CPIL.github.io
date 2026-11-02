import numpy as np

def verify_decomposition(U, V, W, n, m, p):
    rank = U.shape[1]
    # T[i, j, k, l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[l*n+i, c]
    # We want T to equal the matrix multiplication tensor.
    # MM tensor: T[i, j, k, l] = 1 if i==l and j==k else 0? 
    # No, C[i, l] = sum_j A[i, j] B[j, l].
    # Tensor form: sum_c U[i,j,c] V[j,k,c] W[l,i,c] = delta_{j,k} delta_{i,l}?
    # Actually, the tensor is T_{i, j, k, l} such that C_{i, l} = sum_j A_{i, j} B_{j, l}.
    # This corresponds to T_{i, j, k, l} = 1 if j==k and i==l? No.
    # The bilinear form is sum_{i,j,k,l} A_{i,j} B_{j,k} C_{l,i}?
    # Standard: <A, B, C> = sum A_{i,j} B_{j,k} C_{k,i} (trace convention).
    # Task says: "output factor follows the official trace convention W[k*n+i,r]".
    # So W index is k*n + i.
    # Sum over c: U[i, j, c] V[j, k, c] W[k, i, c].
    # This equals 1 if j==k and i==i?
    # Let's check: sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # If we set A, B, C to basis matrices, it should yield 1 if indices match.
    
    # Let's just check if the tensor matches the identity.
    # Tensor T has shape (n*m, m*p, p*n).
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    # Expected tensor:
    # T[i, j, k] where i is A idx (i, j), k is B idx (j, l)?
    # U shape (n*m, rank). Index i in U corresponds to A[i//m, i%m].
    # V shape (m*p, rank). Index j in V corresponds to B[j//p, j%p].
    # W shape (p*n, rank). Index k in W corresponds to C[k//n, k%n].
    
    # We want: sum_c U[a, c] V[b, c] W[c_idx, c] = 1 if A_idx, B_idx, C_idx compatible.
    # Compatibility: A[i, j] * B[j, k] contributes to C[i, k].
    # So U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should sum to 1.
    # Otherwise 0.
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] B[j,k] -> C[i,k]
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T_expected[a_idx, b_idx, c_idx] = 1
                
    if np.allclose(T, T_expected):
        return True
    else:
        return False

# Reconstruct the current program's matrices
DIMENSIONS = [2, 2, 12]
n, m, p = DIMENSIONS
num_pairs = p // 2
rank = num_pairs * 7

U = [[0] * rank for _ in range(n * m)]
V = [[0] * rank for _ in range(m * p)]
W = [[0] * rank for _ in range(p * n)]

for t in range(num_pairs):
    base = t * 7
    for s in range(7):
        c = base + s
        if s == 0:
            U[0][c] = 1
            U[3][c] = 1
            V[0*p + 2*t][c] = 1
            V[1*p + 2*t+1][c] = 1
            W[2*t*n + 0][c] = 1
            W[(2*t+1)*n + 1][c] = 1
        elif s == 1:
            U[2][c] = 1
            U[3][c] = 1
            V[0*p + 2*t][c] = 1
            W[2*t*n + 1][c] = 1
            W[(2*t+1)*n + 1][c] = -1
        elif s == 2:
            U[0][c] = 1
            V[0*p + 2*t+1][c] = 1
            V[1*p + 2*t+1][c] = -1
            W[(2*t+1)*n + 0][c] = 1
            W[(2*t+1)*n + 1][c] = 1
        elif s == 3:
            U[3][c] = 1
            V[1*p + 2*t][c] = 1
            V[0*p + 2*t][c] = -1
            W[2*t*n + 0][c] = 1
            W[2*t*n + 1][c] = 1
        elif s == 4:
            U[0][c] = 1
            U[1][c] = 1
            V[1*p + 2*t+1][c] = 1
            W[2*t*n + 0][c] = -1
            W[(2*t+1)*n + 0][c] = 1
        elif s == 5:
            U[2][c] = 1
            U[0][c] = -1
            V[0*p + 2*t][c] = 1
            V[0*p + 2*t+1][c] = 1
            W[(2*t+1)*n + 1][c] = 1
        elif s == 6:
            U[1][c] = 1
            U[3][c] = -1
            V[1*p + 2*t][c] = 1
            V[1*p + 2*t+1][c] = 1
            W[2*t*n + 0][c] = 1

U = np.array(U)
V = np.array(V)
W = np.array(W)

print(verify_decomposition(U, V, W, n, m, p))
