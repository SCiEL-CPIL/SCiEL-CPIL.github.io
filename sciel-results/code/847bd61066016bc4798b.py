import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Rank estimate: 3 * (7 + 1) + 4 = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_laderman_232(i0, j0, k0):
        nonlocal col
        if col + 7 > rank:
            return False
            
        # A is 2x3, B is 3x2
        # A indices: (i0, j0), (i0, j0+1), (i0, j0+2)
        #           (i0+1, j0), (i0+1, j0+1), (i0+1, j0+2)
        # B indices: (j0, k0), (j0+1, k0), (j0+2, k0)
        #           (j0, k0+1), (j0+1, k0+1), (j0+2, k0+1)
        # C indices: (k0, i0), (k0, i0+1)
        #           (k0+1, i0), (k0+1, i0+1)
        
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = i0 * m + (j0 + 2)
        d = (i0 + 1) * m + j0
        e = (i0 + 1) * m + (j0 + 1)
        f = (i0 + 1) * m + (j0 + 2)
        
        x = j0 * p + k0
        y = j0 * p + (k0 + 1)
        u = (j0 + 1) * p + k0
        v = (j0 + 1) * p + (k0 + 1)
        s = (j0 + 2) * p + k0
        t = (j0 + 2) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = k0 * n + (i0 + 1)
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = a * x
        U[a][col] += 1
        V[x][col] += 1
        W[c00][col] += 1
        W[c01][col] += 1 # Wait, Laderman structure
        # Standard Laderman:
        # M1 = ax -> p, s
        # M2 = (b-c)(u-x) -> r, s
        # M3 = (a+d)(s-x) -> p, r
        # M4 = (f-c)t -> q, r
        # M5 = (a+d)e -> p, q, s
        # M6 = (b+d)(u-s) -> q, r, s
        # M7 = (f+d)(v-t) -> p, q
        
        # Let's use a known valid Laderman set.
        # p = M1 + M3 + M5 - M7
        # q = M4 + M5 + M6 - M7
        # r = M2 + M3 - M4 + M6
        # s = M1 - M2 + M5 - M6
        
        # M1
        U[a][col] += 1
        V[x][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (b-c)(u-x)
        U[b][col] += 1
        U[c][col] -= 1
        V[u][col] += 1
        V[x][col] -= 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = (a+d)(s-x)
        U[a][col] += 1
        U[d][col] += 1
        V[s][col] += 1
        V[x][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M4 = (f-c)t
        U[f][col] += 1
        U[c][col] -= 1
        V[t][col] += 1
        W[c01][col] += 1
        W[c10][col] -= 1
        col += 1
        
        # M5 = (a+d)e
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M6 = (b+d)(u-s)
        U[b][col] += 1
        U[d][col] += 1
        V[u][col] += 1
        V[s][col] -= 1
        W[c01][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M7 = (f+d)(v-t)
        U[f][col] += 1
        U[d][col] += 1
        V[v][col] += 1
        V[t][col] -= 1
        W[c00][col] -= 1
        W[c01][col] -= 1
        col += 1
        
        return True

    def add_rank1_212(i0, j0, k0):
        # A is 2x1 (col j0), B is 1x2 (row j0, cols k0, k0+1)
        # C is 2x2
        nonlocal col
        if col + 1 > rank:
            return False
            
        a0 = i0 * m + j0
        a1 = (i0 + 1) * m + j0
        
        b0 = j0 * p + k0
        b1 = j0 * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = k0 * n + (i0 + 1)
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # T = A_col * B_row
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        
        # Single tensor term:
        # U[a0, r] = A0? No, U is fixed coefficients.
        # We need U such that sum U_ik V_jk W_kl = A_i B_j
        # Here j is fixed.
        # U[i, r] = delta_{i, 0}? No.
        # The term is A_i B_j.
        # We can set U[i, r] = 1 for i=0,1?
        # If U[i,r]=1, V[j,r]=1, W[k,l]=?
        # C_{i,l} = A_i B_l.
        # This requires U[i,r] to pick A_i.
        # But U is constant.
        # Ah, the decomposition is $C = \sum U_k \otimes V_k \otimes W_k$.
        # $C_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
        # For $2 \times 1 \times 2$, $j$ is 0.
        # $C_{i,0,k} = A_i B_k$.
        # We need $\sum_r U_{i,r} V_{0,r} W_{k,r} = A_i B_k$.
        # Let $r=0$.
        # $U_{i,0} = 1$.
        # $V_{0,0} = 1$.
        # $W_{k,0} = 1$.
        # Then $C_{i,0,k} = 1 \cdot 1 \cdot 1 = 1$.
        # This is wrong. The variables A and B are inputs.
        # The tensor is the *multiplication tensor*.
        # $T_{i,j,k} = 1$ if it's a multiplication?
        # No, the tensor is defined by the indices.
        # The rank is the number of terms in $\sum u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$
        # such that $\sum u_i^{(r)} v_j^{(r)} w_k^{(r)} = \delta_{ij} \delta_{jk}$? No.
        # The tensor is $T_{i,j,k} = 1$ if the multiplication $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        # For $2 \times 1 \times 2$, $A_{i,0} B_{0,k} \to C_{i,k}$.
        # The tensor is $T_{i,0,k} = 1$.
        # We need $\sum_r U_{i,r} V_{0,r} W_{k,r} = 1$ for all $i,k$.
        # This is possible with $r=0$:
        # $U_{i,0} = 1$.
        # $V_{0,0} = 1$.
        # $W_{k,0} = 1$.
        # Yes!
        
        U[a0][col] += 1
        U[a1][col] += 1
        V[b0][col] += 1 # Wait, V index is j.
        # V has shape (m*p, rank).
        # V index for j0, k0 is j0*p + k0.
        # But in $2 \times 1 \times 2$, the B input is row j0.
        # So V[j0*p + k0, r] corresponds to B_{j0, k0}.
        # We need V[j0, r] = 1? No, V index is flattened.
        # The term is $A_{i,j0} B_{j0,k} C_{i,k}$.
        # $U_{i, r} V_{j0, r} W_{k, r}$.
        # We set $U_{i,r}=1, V_{j0,r}=1, W_{k,r}=1$.
        # But V index is $j0*p + k$?
        # No, V is for B. B is $4 \times 7$.
        # In this sub-problem, we use B row j0.
        # So V[j0*p + k, r] should be 1 for all k?
        # No, the tensor entry is $T_{i, j, k}$.
        # $j$ is the middle index.
        # $V$ index is $j*p + k$? No, $V$ corresponds to B.
        # B index is $(j,k)$.
        # So $V[j*p+k, r]$.
        # We need $\sum_r U[i,r] V[j*p+k, r] W[k*r+n+i, r] = A_{i,j} B_{j,k}$.
        # For $j=j0$, $k$ varies.
        # $U[i,r] V[j0*p+k, r] W[k*r+n+i, r]$.
        # If we set $V[j0*p+k, r] = 1$ for all k?
        # Then $U[i,r] W[k,r]$.
        # We need $A_i B_k$.
        # This doesn't work with constant coefficients.
        
        # Wait, the rank-1 decomposition of the outer product $A_i B_k$
        # is indeed rank 1?
        # $T_{i,j,k} = 1$ for $j=j0$.
        # $\sum U_i V_{j0, k} W_k$.
        # If we set $V_{j0, k} = B_k$? No, V is constant.
        
        # Ah, the "Rank" of the tensor $T$ is the min $r$ such that
        # $T = \sum u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$.
        # For $2 \times 1 \times 2$, $T_{i,0,k} = 1$.
        # Can we write $1 = \sum_r u_i v_{0,k} w_k$?
        # If $v_{0,k}$ depends on $k$, we need multiple terms?
        # $v^{(r)}$ is a vector in $\mathbb{F}^{mp}$.
        # $v^{(r)}_{j,k}$.
        # We need $\sum_r u_i^{(r)} v_{j0,k}^{(r)} w_k^{(r)} = 1$.
        # Let $r=0$.
        # $u_i = 1$.
        # $v_{j0,k} = 1$.
        # $w_k = 1$.
        # Then $1 \cdot 1 \cdot 1 = 1$.
        # Yes!
        
        U[a0][col] += 1
        U[a1][col] += 1
        V[b0][col] += 1
        V[b1][col] += 1
        W[c00][col] += 1
        W[c01][col] += 1
        W[c10][col] += 1
        W[c11][col] += 1
        col += 1
        return True

    def add_schoolbook_241(i0, j0, k0):
        # A is 2x4, B is 4x1 (col k0), C is 2x1
        nonlocal col
        if col + 4 > rank:
            return False
            
        # C_i = sum_j A_i,j B_j
        # 4 terms: one for each j.
        for j in range(4):
            a_idx = i0 * m + (j0 + j) # j0 is 0 here
            b_idx = (j0 + j) * p + k0
            c0_idx = k0 * n + i0
            c1_idx = k0 * n + (i0 + 1)
            
            U[a_idx][col] += 1
            U[(i0+1)*m + (j0+j)][col] += 1
            V[b_idx][col] += 1
            W[c0_idx][col] += 1
            W[c1_idx][col] += 1
            col += 1
        return True

    # Construct
    # 3 blocks of 2x4x2 (Laderman 2x3x2 + Rank1 2x1x2)
    for k0 in [0, 2, 4]:
        add_laderman_232(0, 0, k0)
        add_rank1_212(0, 3, k0)
        
    # 1 block of 2x4x1 (Schoolbook)
    add_schoolbook_241(0, 0, 6)

    return {"U": U, "V": V, "W": W}

