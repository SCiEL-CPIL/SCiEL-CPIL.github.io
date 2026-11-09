import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # U: (n*m, R) = (6, R)
    # V: (m*p, R) = (21, R)
    # W: (p*n, R) = (14, R)
    
    # We construct a rank 39 decomposition.
    R = 39
    
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    r = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal r
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val
        r += 1

    # Process 3 pairs of columns (0,1), (2,3), (4,5)
    for pair in range(3):
        k0 = 2 * pair
        k1 = 2 * pair + 1
        
        # Strassen terms for 2x2 submatrix (A[0:2, 0:2] * B[0:2, k0:k1])
        # A indices: i in {0,1}, j in {0,1} -> u_idx = i*m + j
        # B indices: j in {0,1}, k in {k0, k1} -> v_idx = j*p + k
        
        # M1 = (A00+A11)(B00+B11)
        add_term(0*m+0, 1, 0*p+k0, 1, k0*n+0, 1)
        add_term(1*m+1, 1, 1*p+k1, 1, k0*n+0, 1) # Wait, W is shared? No, each term is separate.
        # The term is U*V*W.
        # We need to represent the product.
        # My add_term function adds to the current column r.
        # So for M1, we need U to have 1 at A00 and A11? No, U is a vector.
        # The term is sum_r U_r V_r W_r.
        # If I use one rank-1 term, U, V, W are vectors.
        # For M1 = (A00+A11)(B00+B11), this is not a single rank-1 term in the original tensor.
        # It is a bilinear form.
        # To represent $(A00+A11)(B00+B11)$, I need to expand it?
        # No, the decomposition is a sum of rank-1 tensors.
        # $T = \sum_r u_r \otimes v_r \otimes w_r$.
        # If I want to compute $M_1$, I can't just put 1 in U and V.
        # $M_1$ is an intermediate value.
        # The final output is a linear combination of $M_1 \dots M_7$.
        # So the decomposition is:
        # $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$
        # For Strassen, we define 7 "multiplications" $M_s$.
        # Each $M_s$ is a product of a linear combo of A and a linear combo of B.
        # Let $L_s(A)$ be the linear combo of A, $R_s(B)$ be the linear combo of B.
        # $M_s = L_s(A) R_s(B)$.
        # This $M_s$ can be represented as a rank-1 term if we consider the tensor of $L_s$ and $R_s$.
        # But $L_s(A)$ is a vector, $R_s(B)$ is a vector.
        # The product $L_s(A) R_s(B)$ is a scalar.
        # This scalar is then added to $C_{i,k}$ with coefficient $W_{k,i,r}$.
        # So, for each Strassen step $s$, we create a rank-1 term where:
        # $U_{u, r} = $ coefficient of $A_u$ in $L_s(A)$
        # $V_{v, r} = $ coefficient of $B_v$ in $R_s(B)$
        # $W_{w, r} = $ coefficient of $M_s$ in $C_w$
        
        # So, for M1:
        # $L_1(A) = A_{0,0} + A_{1,1}$
        # $R_1(B) = B_{0,k0} + B_{1,k1}$
        # $M_1$ contributes to:
        # $C_{0,k0}$ with coeff 1
        # $C_{0,k1}$ with coeff 1
        # $C_{1,k0}$ with coeff 1
        # $C_{1,k1}$ with coeff 1
        # Wait, the reconstruction is:
        # $C_{0,k0} = M_1 + M_4 - M_2 + M_6$
        # $C_{0,k1} = M_1 + M_3$
        # $C_{1,k0} = M_2 + M_7$
        # $C_{1,k1} = M_1 - M_3 + M_5 + M_7$
        
        # So for r corresponding to M1:
        # U[0][r] = 1 (A00)
        # U[3][r] = 1 (A11) -- index 1*m+1 = 4? No, m=3. A11 is i=1, j=1. idx = 1*3+1 = 4.
        # U[4][r] = 1
        # V[0*p+k0][r] = 1 (B0, k0)
        # V[1*p+k1][r] = 1 (B1, k1)
        # W[k0*n+0][r] = 1
        # W[k1*n+0][r] = 1
        # W[k0*n+1][r] = 1
        # W[k1*n+1][r] = 1
        
        # Let's implement this properly.
        
        # M1
        U[0*m+0][r] = 1
        U[1*m+1][r] = 1
        V[0*p+k0][r] = 1
        V[1*p+k1][r] = 1
        W[k0*n+0][r] = 1
        W[k1*n+0][r] = 1
        W[k0*n+1][r] = 1
        W[k1*n+1][r] = 1
        r += 1
        
        # M2 = (A10+A11) B0,k0
        U[1*m+0][r] = 1
        U[1*m+1][r] = 1
        V[0*p+k0][r] = 1
        W[k0*n+1][r] = 1
        r += 1
        
        # M3 = A00 (B1,k0 - B1,k1)
        U[0*m+0][r] = 1
        V[1*p+k0][r] = 1
        V[1*p+k1][r] = -1
        W[k1*n+0][r] = 1
        r += 1
        
        # M4 = A11 (B0,k1 - B0,k0)
        U[1*m+1][r] = 1
        V[0*p+k1][r] = 1
        V[0*p+k0][r] = -1
        W[k0*n+0][r] = 1
        r += 1
        
        # M5 = (A00+A01) B1,k1
        U[0*m+0][r] = 1
        U[0*m+1][r] = 1
        V[1*p+k1][r] = 1
        W[k1*n+1][r] = 1
        r += 1
        
        # M6 = (A10-A00) (B0,k0+B0,k1)
        U[1*m+0][r] = 1
        U[0*m+0][r] = -1
        V[0*p+k0][r] = 1
        V[0*p+k1][r] = 1
        W[k0*n+0][r] = 1
        r += 1
        
        # M7 = (A01-A11) (B1,k0+B1,k1)
        U[0*m+1][r] = 1
        U[1*m+1][r] = -1
        V[1*p+k0][r] = 1
        V[1*p+k1][r] = 1
        W[k0*n+1][r] = 1
        W[k1*n+1][r] = 1
        r += 1
        
        # Direct terms for j=2
        # C0,k0 += A02 B2,k0
        U[0*m+2][r] = 1
        V[2*p+k0][r] = 1
        W[k0*n+0][r] = 1
        r += 1
        
        # C0,k1 += A02 B2,k1
        U[0*m+2][r] = 1
        V[2*p+k1][r] = 1
        W[k1*n+0][r] = 1
        r += 1
        
        # C1,k0 += A12 B2,k0
        U[1*m+2][r] = 1
        V[2*p+k0][r] = 1
        W[k0*n+1][r] = 1
        r += 1
        
        # C1,k1 += A12 B2,k1
        U[1*m+2][r] = 1
        V[2*p+k1][r] = 1
        W[k1*n+1][r] = 1
        r += 1

    # Last column k=6
    # C0,6 = A00 B0,6 + A01 B1,6 + A02 B2,6
    U[0*m+0][r] = 1
    V[0*p+6][r] = 1
    W[6*n+0][r] = 1
    r += 1
    
    U[0*m+1][r] = 1
    V[1*p+6][r] = 1
    W[6*n+0][r] = 1
    r += 1
    
    U[0*m+2][r] = 1
    V[2*p+6][r] = 1
    W[6*n+0][r] = 1
    r += 1
    
    # C1,6 = A10 B0,6 + A11 B1,6 + A12 B2,6
    U[1*m+0][r] = 1
    V[0*p+6][r] = 1
    W[6*n+1][r] = 1
    r += 1
    
    U[1*m+1][r] = 1
    V[1*p+6][r] = 1
    W[6*n+1][r] = 1
    r += 1
    
    U[1*m+2][r] = 1
    V[2*p+6][r] = 1
    W[6*n+1][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
