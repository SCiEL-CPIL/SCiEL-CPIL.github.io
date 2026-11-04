"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-48 decomposition using a "split" structure:
    # Treat the 4x4 matrix as two 2x2 blocks horizontally (cols 0-1, 2-3).
    # For each row i (0..2) and each block b (0..1), compute the 1x2 @ 2x2 product.
    # The rank of 1x2 @ 2x2 is 3 (not 4).
    # Total rank = 3 * 2 * 3 = 18? No, 3 (n) * 2 (blocks) * 3 (rank per block) = 18.
    # Wait, let's verify rank(1x2 @ 2x2).
    # Tensor T in U(1) x V(2) x W(2). T_{a, b, c} = 1 if a=0 and b=c.
    # This is the identity matrix viewed as a tensor.
    # Rank of I_2 is 2? No.
    # T_{0, 0, 0}=1, T_{0, 1, 1}=1.
    # Decomposition:
    # U_{0, 1}=1, V_{0, 1}=1, W_{0, 1}=1
    # U_{0, 2}=1, V_{1, 2}=1, W_{1, 2}=1
    # Rank 2?
    # Let's check if rank 2 is possible for 1x2 @ 2x2.
    # The tensor is $e_0 \otimes (e_0 \otimes e_0 + e_1 \otimes e_1)$.
    # This is a symmetric tensor in the V-W space, multiplied by $e_0$ in U.
    # The rank of a tensor $u \otimes A$ where $A$ is a matrix is the rank of $A$.
    # Rank of $I_2$ is 2.
    # So rank(1x2 @ 2x2) is 2.
    # Therefore, total rank = 3 (n) * 2 (blocks) * 2 (rank per block) = 12.
    
    # Let's double check this.
    # Multiplication of 1x2 by 2x2.
    # Input x = [x0, x1], Matrix Y = [[y00, y01], [y10, y11]].
    # Output z = [x0*y00 + x1*y10, x0*y01 + x1*y11].
    # Tensor T_{i,j,k} where i is U-index (1), j is V-index (2), k is W-index (2).
    # T_{0, j, k} = 1 if j=k, else 0.
    # This corresponds to the matrix $I_2$ in the (j,k) plane.
    # The rank of this tensor is the rank of the matrix $I_2$, which is 2.
    # Decomposition:
    # Term 1: U[0]=1, V[0]=1, W[0]=1. (Contributes to z[0] from x[0]y[0])
    # Term 2: U[0]=1, V[1]=1, W[1]=1. (Contributes to z[1] from x[1]y[1])
    # Check:
    # z[0] = U[0]*V[0]*W[0]*x[0]*y[0] + ... = 1*1*1*x[0]*y[00] + 1*1*1*x[0]*y[01]? No.
    # The indices are:
    # U index: 0 (since n=1)
    # V index: row of Y (0 or 1)
    # W index: col of Y (0 or 1)
    # Output index k (col of Z) is the W index.
    # Input index j (row of Y) is the V index.
    # Input index i (col of X) is the U index.
    # Wait, standard convention:
    # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Here A is 1x2, B is 2x2.
    # i=0. j in 0..1. k in 0..1.
    # T_{0,j,k} = 1 if A_{0,j} and B_{j,k} are the "basis" elements.
    # Actually, the tensor defines the multiplication map.
    # The term is $A_{i,j} B_{j,k}$.
    # For fixed i=0, the coefficient of $A_{0,j} B_{j,k}$ is 1.
    # So $T_{0,j,k} = 1$.
    # This is NOT the identity matrix. It is the all-ones matrix $J_2$ in the (j,k) plane?
    # No, $A_{0,j}$ is a specific element. $B_{j,k}$ is a specific element.
    # The product term is $A_{0,j} B_{j,k}$.
    # The coefficient of the monomial $A_{0,j} B_{j,k}$ in the sum is 1.
    # So the tensor entry $T_{(i,j), (j,k), ?}$...
    # Let's stick to the factor matrices definition.
    # $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$? No.
    # Standard form: $C_{i,k} = \sum_{j} \sum_{r} U_{i,j, r} V_{j,k, r}$? No.
    # The task specifies shapes:
    # U: (n*m, rank) -> U[i, r] where i = i_row * m + j_col? Or i_row + j_col*m?
    # Current code: `U[i * m + j]`. So index is `i*m + j`.
    # V: (m*p, rank) -> V[j, r] where index is `j*p + k`.
    # W: (p*n, rank) -> W[k, r] where index is `k*n + i`.
    # Formula: $C_{i,k} = \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$?
    # No, the sum is over j as well?
    # Let's look at the schoolbook code:
    # `U[i * m + j][column] = 1`
    # `V[j * p + k][column] = 1`
    # `W[k * n + i][column] = 1`
    # This creates a term for each (i,j,k).
    # The reconstructed value at output (i,k) would be:
    # $\sum_{j} \sum_{r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$?
    # In the schoolbook case, for a fixed r (which corresponds to a specific i,j,k),
    # $U_{i*m+j, r}=1$, $V_{j*p+k, r}=1$, $W_{k*n+i, r}=1$.
    # If we sum over r, we get 1 for the specific (i,j,k) combination.
    # But the output $C_{i,k}$ should be $\sum_j A_{i,j} B_{j,k}$.
    # The linear combination of the monomials $A_{i,j} B_{j,k}$ is $\sum_j 1 \cdot A_{i,j} B_{j,k}$.
    # The factorization allows us to compute this sum efficiently.
    # The "rank" is the number of multiplications (terms).
    #
    # Back to 1x2 @ 2x2.
    # We want to compute $C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}$
    # and $C_{0,1} = A_{0,0} B_{0,1} + A_{0,1} B_{1,1}$.
    # Using rank 2:
    # Term 1: r=0.
    #   We need to generate $A_{0,0} B_{0,0}$ for C00 and $A_{0,0} B_{0,1}$ for C01?
    #   Let's try to isolate $A_{0,0}$ and $A_{0,1}$.
    #   If we use:
    #   r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. (Monomial $A_{0,0} B_{0,0}$)
    #   r=1: U[0,1]=1, V[0,1]=1, W[1,1]=1. (Monomial $A_{0,0} B_{0,1}$)
    #   r=2: U[0,2]=1, V[1,2]=1, W[0,2]=1. (Monomial $A_{0,1} B_{1,0}$)
    #   r=3: U[0,3]=1, V[1,3]=1, W[1,3]=1. (Monomial $A_{0,1} B_{1,1}$)
    #   This is rank 4.
    #
    # Can we do rank 3?
    # Strassen-like for 1x2 @ 2x2?
    # Let $X_0 = A_{0,0}$, $X_1 = A_{0,1}$.
    # Let $Y_{00}, Y_{01}, Y_{10}, Y_{11}$.
    # $C_{0,0} = X_0 Y_{00} + X_1 Y_{10}$
    # $C_{0,1} = X_0 Y_{01} + X_1 Y_{11}$
    # Try:
    # $P_1 = X_0 (Y_{00} + Y_{01})$
    # $P_2 = X_1 (Y_{10} + Y_{11})$
    # $P_3 = (X_0 + X_1) (Y_{00} + Y_{10})$?
    # $C_{0,0} = P_1 - X_0 Y_{01} + ...$ doesn't seem to work simply.
    #
    # Actually, rank(1x2 @ 2x2) is known to be 3.
    # Reference: "The tensor rank of matrix multiplication"
    # For $1 \times 2 \times 2$, rank is 3.
    # Decomposition:
    # $U_1 = 1, V_1 = 1, W_1 = 1$ ? No, these are scalars in the decomposition?
    # The factors are matrices.
    #
    # Let's construct the rank 3 decomposition for 1x2 @ 2x2.
    # Indices: i=0 (U row), j=0,1 (V row), k=0,1 (W col).
    # U is 1x3. V is 2x3. W is 2x3.
    #
    # Let's try:
    # r=0: U[0]=1. V[0]=1, V[1]=0. W[0]=1, W[1]=1.
    #       Term: $1 * (A_{0,0} * 1 + A_{0,1} * 0) * (1*B_{0,0} + 1*B_{0,1}) = A_{0,0}(B_{0,0}+B_{0,1})$
    # r=1: U[0]=1. V[0]=0, V[1]=1. W[0]=1, W[1]=1.
    #       Term: $1 * (A_{0,0} * 0 + A_{0,1} * 1) * (1*B_{1,0} + 1*B_{1,1}) = A_{0,1}(B_{1,0}+B_{1,1})$
    # r=2: U[0]=1. V[0]=1, V[1]=1. W[0]=1, W[1]=-1.
    #       Term: $1 * (A_{0,0} + A_{0,1}) * (1*B_{0,0} - 1*B_{1,0}) = (A_{0,0}+A_{0,1})(B_{0,0}-B_{1,0})$
    #
    # Sum = $A_{0,0}B_{0,0} + A_{0,0}B_{0,1} + A_{0,1}B_{1,0} + A_{0,1}B_{1,1} + (A_{0,0}+A_{0,1})(B_{0,0}-B_{1,0})$
    #     = $A_{0,0}B_{0,0} + A_{0,0}B_{0,1} + A_{0,1}B_{1,0} + A_{0,1}B_{1,1} + A_{0,0}B_{0,0} - A_{0,0}B_{1,0} + A_{0,1}B_{0,0} - A_{0,1}B_{1,0}$
    # This has extra terms and wrong coefficients.
    #
    # Correct Rank 3 for 1x2 @ 2x2:
    # $P_1 = A_0 (B_{00} + B_{01})$
    # $P_2 = A_1 (B_{10} + B_{11})$
    # $P_3 = (A_0 + A_1) (B_{00} + B_{10})$? No.
    #
    # Let's use the known decomposition:
    # $U_1 = [1], V_1 = [1, 0], W_1 = [1, 1]$ -> $A_0 (B_{00} + B_{01})$
    # $U_2 = [1], V_2 = [0, 1], W_2 = [1, 1]$ -> $A_1 (B_{10} + B_{11})$
    # $U_3 = [1], V_3 = [1, 1], W_3 = [1, -1]$ -> $(A_0+A_1) (B_{00} - B_{10})$? No.
    #
    # Let's solve for coefficients.
    # We want $\sum_r U_{0,r} V_{j,r} W_{k,r} = \delta_{jk}$? No, the target is $\delta_{j, \text{something}}$?
    # The target tensor is $T_{0,j,k} = \delta_{j,k}$? No.
    # The target is $C_k = \sum_j A_j B_{j,k}$.
    # So $T_{j,k}$ (coeff of $A_j B_{j,k}$) is 1 if we fix j?
    # No, $A_j$ is fixed, $B_{j,k}$ varies.
    # The term is $A_j B_{j,k}$.
    # So for each j, we sum over k? No.
    # The output is a vector $C$ of size 2.
    # $C_k = \sum_j A_j B_{j,k}$.
    # So the coefficient of $A_j B_{j,k}$ in $C_k$ is 1.
    # This means the tensor is $T_{j,k, \text{out}_k} = 1$.
    # Wait, the output index is k.
    # So $T_{j,k} = 1$ for all j,k?
    # No. $C_0 = A_0 B_{0,0} + A_1 B_{1,0}$.
    # $C_1 = A_0 B_{0,1} + A_1 B_{1,1}$.
    # So the coefficient of $A_j B_{j,k}$ is 1.
    # This is the tensor $T_{j,k} = 1$ for all $j \in \{0,1\}, k \in \{0,1\}$.
    # This is the all-ones matrix $J_2$.
    # The rank of $J_2$ is 1!
    # $J_2 = [1, 1; 1, 1] = [1; 1] [1, 1]$.
    # So we can decompose it with rank 1?
    # $P = (A_0 + A_1) (B_{0,0} + B_{0,1} + B_{1,0} + B_{1,1})$?
    # No, the factors must be linear in A and B separately.
    # $U V W$ form.
    # If rank 1: $U=1, V=[v_0, v_1], W=[w_0, w_1]$.
    # Term: $1 \cdot (v_0 A_0 + v_1 A_1) \cdot (w_0 B_{0,0} + w_1 B_{0,1})$? No.
    # The V index is j (row of B). The W index is k (col of B, which is output index).
    # So $V_j$ and $W_k$.
    # Term: $(\sum_j V_j A_j) (\sum_k W_k B_{j,k})$? No, j is fixed for V and W?
    # In the factorization $\sum_r U_{i,r} V_{j,r} W_{k,r}$, the index j is tied to V and k to W.
    # But in the multiplication, j is the sum index.
    # The schoolbook code sums over r, but j is part of the index of U and V.
    # $C_{i,k} = \sum_j \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
    # So for a fixed r, we have a term $U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
    # This term contributes to $C_{i,k}$ only if the indices match.
    # In the schoolbook, for a specific r, j is fixed. So the term is $A_{i,j} B_{j,k}$.
    #
    # For 1x2 @ 2x2, we want to represent $C_0 = A_0 B_{0,0} + A_1 B_{1,0}$ and $C_1 = A_0 B_{0,1} + A_1 B_{1,1}$.
    # This is equivalent to computing $C = A \otimes B$? No.
    # It is $C_k = \sum_j A_j B_{j,k}$.
    # This is a matrix-vector product where A is a row vector and B is a matrix.
    #
    # Can we do it with rank 2?
    # $r=0$: Contribute to $C_0$ and $C_1$ using $A_0$.
    # $r=1$: Contribute to $C_0$ and $C_1$ using $A_1$.
    # $r=0$: $U[0]=1$. $V[0]=1, V[1]=0$. $W[0]=1, W[1]=1$.
    #   Term: $1 \cdot (1 \cdot A_0 + 0 \cdot A_1) \cdot (1 \cdot B_{0,0} + 1 \cdot B_{0,1})$?
    #   Wait, V and W are matrices of size (m*p) x rank and (p*n) x rank.
    #   For 1x2 @ 2x2: n=1, m=2, p=2.
    #   U: 1x2. V: 4x2 (indices j*p+k = 0..3). W: 2x2 (indices k*n+i = 0..1).
    #   $r=0$:
    #   $U[0] = 1$.
    #   $V[0] = 1, V[1] = 0, V[2] = 0, V[3] = 0$. (Only j=0, k=0 active? No, V index is j*p+k. If j=0, indices 0,1. If j=1, indices 2,3.)
    #   We want to pick up $A_0$. So $V[0]$ and $V[1]$ (j=0) should be non-zero?
    #   Actually, $V_{j*p+k, r}$ multiplies $B_{j,k}$.
    #   If we set $V[0]=1, V[1]=1$ (j=0, k=0 and k=1), then we get $B_{0,0} + B_{0,1}$.
    #   $W[0]=1, W[1]=1$ (k=0 and k=1).
    #   Term: $1 \cdot (A_0 \cdot 1 + A_1 \cdot 0) \cdot (B_{0,0} \cdot 1 + B_{0,1} \cdot 1)$?
    #   No, the sum is over r. For a fixed r, the value added to $C_{i,k}$ is $U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
    #   But j is summed over?
    #   The formula is $\sum_j \sum_r ...$
    #   So for fixed r, we sum over j.
    #   $r=0$:
    #   $U[0]=1$.
    #   $V[0]=1, V[1]=1$ (j=0). $V[2]=0, V[3]=0$ (j=1).
    #   $W[0]=1, W[1]=1$.
    #   Contribution to $C_0$ (k=0): $\sum_j U[0] V[j*2+0] W[0*1+0] = 1 \cdot (V[0]*1 + V[2]*1) \cdot 1 = 1 \cdot (1+0) \cdot 1 = 1$.
    #   Wait, $W$ index is $k*n+i = 0*1+0=0$. So $W[0]=1$.
    #   Contribution to $C_1$ (k=1): $\sum_j U[0] V[j*2+1] W[1*1+0] = 1 \cdot (V[1]*1 + V[3]*1) \cdot 1 = 1 \cdot (1+0) \cdot 1 = 1$.
    #   So this term gives $1 \cdot A_0 \cdot (B_{0,0} + B_{0,1})$?
    #   No, the "value" is the coefficient. The actual multiplication is $A_{i,j} B_{j,k}$.
    #   The coefficient of $A_0 B_{0,0}$ in $C_0$ is 1.
    #   The coefficient of $A_0 B_{0,1}$ in $C_0$ is 0?
    #   My calculation: $C_0$ gets 1. $C_1$ gets 1.
    #   This corresponds to $A_0 (B_{0,0} + B_{0,1})$.
    #   This contributes $A_0 B_{0,0}$ to $C_0$ and $A_0 B_{0,1}$ to $C_1$.
    #   This is correct for the $A_0$ part!
    #
    # $r=1$:
    #   $U[0]=1$.
    #   $V[2]=1, V[3]=1$ (j=1). $V[0]=0, V[1]=0$.
    #   $W[0]=1, W[1]=1$.
    #   Contribution to $C_0$: $\sum_j U[0] V[j*2+0] W[0] = 1 \cdot (V[0] + V[2]) \cdot 1 = 1 \cdot (0+1) \cdot 1 = 1$.
    #   Contribution to $C_1$: $\sum_j U[0] V[j*2+1] W[1] = 1 \cdot (V[1] + V[3]) \cdot 1 = 1 \cdot (0+1) \cdot 1 = 1$.
    #   This corresponds to $A_1 (B_{1,0} + B_{1,1})$.
    #   This contributes $A_1 B_{1,0}$ to $C_0$ and $A_1 B_{1,1}$ to $C_1$.
    #   This is correct for the $A_1$ part!
    #
    # So rank 2 works for 1x2 @ 2x2!
    #
    # Total rank for 3x4 @ 4x4:
    # Split 4x4 into two 2x2 blocks horizontally.
    # For each i (0..2), we have a 1x2 @ 2x2 block.
    # Rank per block = 2.
    # Total rank = 3 * 2 * 2 = 12.
    
    n, m, p = 3, 4, 4
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for b in range(2): # 2 blocks
            # Block b uses cols 2*b and 2*b+1
            # r0: for A_i (j=0 in block, i.e., global j=2*b)
            # r1: for A_i (j=1 in block, i.e., global j=2*b+1)
            
            # r0
            r0 = col
            # U index: i*m + j. j=2*b.
            U[i * m + 2*b][r0] = 1
            # V index: j*p + k. j=2*b. k=0,1.
            V[(2*b) * p + 0][r0] = 1
            V[(2*b) * p + 1][r0] = 1
            # W index: k*n + i. k=0,1.
            W[0 * n + i][r0] = 1
            W[1 * n + i][r0] = 1
            col += 1
            
            # r1
            r1 = col
            # U index: i*m + j. j=2*b+1.
            U[i * m + (2*b+1)][r1] = 1
            # V index: j*p + k. j=2*b+1. k=0,1.
            V[(2*b+1) * p + 0][r1] = 1
            V[(2*b+1) * p + 1][r1] = 1
            # W index: k*n + i. k=0,1.
            W[0 * n + i][r1] = 1
            W[1 * n + i][r1] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
