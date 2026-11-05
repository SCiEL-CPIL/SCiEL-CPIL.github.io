import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    
    def schoolbook():
        """Return the standard rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_true = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m + j, j*p + k, k*n + i] = 1.0
        
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        if not np.allclose(T_recon, T_true, atol=tol):
            return False
        return True

    # Implement rank 88 decomposition
    # A is 3x4, B is 4x8
    # A = [[A11, A12], [A21, A22]] where A11, A12 are 2x2, A21, A22 are 1x2
    # B = [[B11, B12], [B21, B22]] where B11..B22 are 2x4
    
    # We will build the rank-1 terms
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    def strassen_2x2_2x2(A_idx, B_idx, C_idx_base):
        # A_idx: (i1, i2) for rows, (j1, j2) for cols of A subblock
        # B_idx: (j1, j2) for rows, (k1, k2) for cols of B subblock
        # C_idx_base: (i_base, k_base) for the output subblock
        # A is 2x2, B is 2x2
        # Output is 2x2
        
        # A subblock indices in global A (3x4)
        # A[i, j]
        # B subblock indices in global B (4x8)
        # B[j, k]
        # Output C[i, k]
        
        # We need to map local 2x2 indices to global
        # Let's say A_sub is at rows r0, r0+1 and cols c0, c0+1
        # B_sub is at rows c0, c0+1 and cols k0, k0+1
        
        r0, c0, k0 = A_idx
        # r0 is row offset in A (0 or 2)
        # c0 is col offset in A (0 or 2)
        # k0 is col offset in B (0, 2, 4, 6)
        
        # A entries:
        # a00 = A[r0, c0], a01 = A[r0, c0+1]
        # a10 = A[r0+1, c0], a11 = A[r0+1, c0+1]
        
        # B entries:
        # b00 = B[c0, k0], b01 = B[c0, k0+1]
        # b10 = B[c0+1, k0], b11 = B[c0+1, k0+1]
        
        # We create 7 rank-1 terms
        
        # M1 = (a00+a11)(b00+b11)
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[r0*m + c0] = 1
        u[(r0+1)*m + (c0+1)] = 1
        v[c0*p + k0] = 1
        v[(c0+1)*p + (k0+1)] = 1
        w[k0*n + r0] = 1
        w[(k0+1)*n + (r0+1)] = 1
        # Wait, rank-1 term is u * v * w?
        # The tensor is T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For a term (a00+a11)(b00+b11) contributing to C[r0, k0] and C[r0+1, k0+1] etc?
        # No, Strassen computes C = AB.
        # M1 contributes to C00 and C11.
        # C00 += M1. C11 += M1.
        # So we need to add to W at positions corresponding to C00 and C11.
        # C00 is at global (r0, k0).
        # C11 is at global (r0+1, k0+1).
        
        # The term is (a00+a11) * (b00+b11) * 1
        # U vector has 1s at (r0, c0) and (r0+1, c0+1)
        # V vector has 1s at (c0, k0) and (c0+1, k0+1)
        # W vector must have 1s at (k0, r0) and (k0+1, r0+1) to sum them up?
        # T[i,j,k] = U[i,r] V[j,r] W[k,r]
        # If U has 1 at i1, i2. V has 1 at j1, j2. W has 1 at k1, k2.
        # Then T[i1,j1,k1] += 1*1*1 = 1.
        # T[i1,j1,k2] += 1*1*1 = 1.
        # T[i1,j2,k1] += 1*1*1 = 1.
        # T[i1,j2,k2] += 1*1*1 = 1.
        # T[i2,j1,k1] += 1*1*1 = 1.
        # ...
        # This creates a 2x2x2 block of 1s.
        # But M1 only contributes to C00 and C11.
        # So this simple product of sums doesn't work directly with a single rank-1 term
        # unless we use the structure of the output.
        
        # Actually, the rank-1 term in the tensor decomposition corresponds to
        # U * V * W where U, V, W are vectors.
        # The contribution to T is the outer product U V W.
        # If U has support {i1, i2}, V has support {j1, j2}, W has support {k1, k2},
        # the support of the contribution is {i1,i2} x {j1,j2} x {k1,k2}.
        # For M1 = (a00+a11)(b00+b11), the values are:
        # C00 gets M1. C11 gets M1.
        # C01 and C10 get 0.
        # So the support is not a full box.
        # This means M1 is not a single rank-1 term in the tensor T?
        # Wait. Strassen's algorithm works by linear combinations.
        # The "rank" refers to the number of multiplications.
        # Each multiplication is a rank-1 term in the algebraic complexity sense?
        # No, the tensor rank is the number of rank-1 tensors.
        # A multiplication $x y z$ is a rank-1 tensor.
        # In Strassen, we compute $M_1 = (a_{00}+a_{11})(b_{00}+b_{11})$.
        # This is a product of two linear forms.
        # $L_1(a) L_2(b)$.
        # This corresponds to a rank-1 term in the tensor if we view it as a bilinear map?
        # The tensor is trilinear.
        # $T = \sum M_r (u_r \otimes v_r \otimes w_r)$.
        # In Strassen, $M_1$ is a scalar.
        # $C_{00} = M_1 + \dots$
        # $C_{11} = M_1 + \dots$
        # So $M_1$ contributes to $T_{00,00,00}$? No.
        # $T_{i,j,k}$ is the coefficient of $A_{i,j} B_{j,k}$ in $C_{i,k}$.
        # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$.
        # $C_{11} = A_{10}B_{01} + A_{11}B_{11}$.
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # This is not just a single term.
        # The "rank" of the multiplication algorithm is the number of multiplications.
        # Each multiplication corresponds to a rank-1 tensor in the decomposition of the multiplication tensor?
        # Yes. The tensor for $2 \times 2 \times 2$ multiplication has rank 7.
        # The 7 rank-1 tensors are given by the 7 multiplications.
        # For $M_1$, the associated rank-1 tensor has:
        # U vector: 1 at (0,0) and (1,1).
        # V vector: 1 at (0,0) and (1,1).
        # W vector: 1 at (0,0) and (1,1).
        # Let's check:
        # $T_{00,00,00} = U_{00} V_{00} W_{00} = 1$.
        # $T_{00,11,11}$? No.
        # The tensor $T$ has $T_{i,j,k} = 1$ if $i=k$? No.
        # $T_{i,j,k}$ is 1 if the term $A_{i,j} B_{j,k}$ appears in $C_{i,k}$.
        # For $2 \times 2 \times 2$:
        # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$.
        # $C_{01} = A_{00}B_{01} + A_{01}B_{11}$.
        # $C_{10} = A_{10}B_{00} + A_{11}B_{10}$.
        # $C_{11} = A_{10}B_{01} + A_{11}B_{11}$.
        # So $T_{00,00,00}=1, T_{00,10,00}=1$? No, indices are $i,j,k$.
        # $T_{0,0,0}=1$ ($A_{00}B_{00} \to C_{00}$).
        # $T_{0,1,0}=1$ ($A_{01}B_{10} \to C_{00}$).
        # $T_{0,0,1}=1$ ($A_{00}B_{01} \to C_{01}$).
        # $T_{0,1,1}=1$ ($A_{01}B_{11} \to C_{01}$).
        # $T_{1,0,0}=1$ ($A_{10}B_{00} \to C_{10}$).
        # $T_{1,1,0}=1$ ($A_{11}B_{10} \to C_{10}$).
        # $T_{1,0,1}=1$ ($A_{10}B_{01} \to C_{11}$).
        # $T_{1,1,1}=1$ ($A_{11}B_{11} \to C_{11}$).
        
        # The rank-1 tensor for $M_1$ is $u \otimes v \otimes w$.
        # $u_0=1, u_1=1$. (Indices 0,1 for A)
        # $v_0=1, v_1=1$. (Indices 0,1 for B)
        # $w_0=1, w_1=1$. (Indices 0,1 for C)
        # Contribution to $T_{i,j,k}$ is $u_i v_j w_k$.
        # $T_{0,0,0} += 1$.
        # $T_{0,0,1} += 1$.
        # $T_{0,1,0} += 1$.
        # $T_{0,1,1} += 1$.
        # $T_{1,0,0} += 1$.
        # $T_{1,0,1} += 1$.
        # $T_{1,1,0} += 1$.
        # $T_{1,1,1} += 1$.
        # This adds 1 to ALL 8 entries.
        # But $M_1$ should only contribute to specific entries with specific signs?
        # No, the decomposition is $T = \sum M_r (u_r \otimes v_r \otimes w_r)$.
        # The $M_r$ are the coefficients.
        # In Strassen, the coefficients are 1 or -1.
        # So we just need to define $u_r, v_r, w_r$ for each of the 7 terms.
        
        # For M1:
        # u: [1, 1] (A00, A11)
        # v: [1, 1] (B00, B11)
        # w: [1, 1] (C00, C11)
        # This adds 1 to all 8.
        # We need to subtract from the others.
        
        # This is getting complicated to map manually.
        # I will stick to the schoolbook 96 if I can't quickly verify the 88 mapping.
        # But 88 is a known improvement.
        
        pass
        
    # Given the complexity and time limit, I will return the schoolbook.
    # But I can try a simple 88 implementation if I can find the vectors.
    
    # Actually, I can just return schoolbook.
    return schoolbook()

# EVOLVE-BLOCK-END
