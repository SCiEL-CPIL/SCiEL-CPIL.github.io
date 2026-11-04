"""Direct construction of the exact tensor decomposition for (2,3,5) matrix multiplication.

We aim for a rank-28 decomposition by exploiting the rectangular structure.
The schoolbook rank is 2*3*5 = 30.
We can save 2 ranks by using a Strassen-like trick on a 2x2 sub-block of the output,
or by using a known bound.

Actually, a simpler and robust approach is to use the "border rank" or known explicit decompositions.
However, for 2x3x5, the rank is known to be 28? 
Let's check: R(2,3,5) <= R(2,3,3) + R(2,3,2).
R(2,3,3) is the rank for 2x3 * 3x3.
R(2,3,2) is the rank for 2x3 * 3x2.

Wait, the standard decomposition splits the third dimension (p).
If we split p=5 into p1=3 and p2=2, we need R(2,3,3) + R(2,3,2).
R(2,3,3): We can view this as 2*(3x3). The rank of 3x3 is 7 (Strassen). So R(2,3,3) <= 2*7 = 14?
No, that's if we treat it as two independent 1x3x3 problems? No.
The tensor is sum over i=1..2, j=1..3, k=1..3.
We can use the fact that R(2,3,3) <= 2 * R(1,3,3) is not quite right because the A matrix has 2 rows.
Actually, R(2,3,3) is the rank of the 2x3x3 tensor.
We can decompose the 2x3 matrix A into two 1x3 rows? No, the multiplication is linear in A.
So R(2,3,3) <= 2 * R(1,3,3).
R(1,3,3) is the rank for 1x3 * 3x3 = 1x3. This is just a linear map from 9-dim to 3-dim. The rank is 3?
No, the tensor T_{1,3,3} has entries T_{i,j,k} = delta_{j,k} (if we consider the identity map on the 3rd dim? No).
For 1x3 * 3x3: C_i = sum_j A_{1,j} B_{j,k}.
This is a linear combination of the rows of B.
The tensor has 1*3*3 = 9 terms in schoolbook.
Can we do better?
R(1,3,3) is the rank of the matrix multiplication tensor for 1x3 and 3x3.
This is equivalent to the rank of the 3x3 identity matrix? No.
It is the rank of the map A -> A*B.
Since A is a vector of length 3, and B is 3x3, the output is a vector of length 3.
The tensor is T_{j,k} = A_j B_{k,j}? No, indices are (row of A, col of A, row of B, col of B).
A is 1x3. B is 3x3. C is 1x3.
C_k = sum_j A_j B_{j,k}.
The tensor entries are T_{j,k} = 1 if the term A_j B_{j,k} exists.
This is a 3x3 tensor (index j for A, index k for C, index j for B... wait, B is indexed by j,k).
The term is A_j * B_{j,k}.
So the tensor is sum_{j,k} A_j B_{j,k} C_k.
This is the identity tensor of size 3? No.
It is the tensor I_3 = sum_{j,k} e_j \otimes e_{j,k} \otimes e_k.
The rank of the identity tensor I_n is n.
So R(1,3,3) = 3.
Thus R(2,3,3) <= 2 * 3 = 6.
Wait, is R(2,3,3) really 6?
Let's check. We have 2 rows of A.
C_{i,k} = sum_j A_{i,j} B_{j,k}.
This is sum_i sum_j A_{i,j} B_{j,k} C_{i,k}.
We can group by i.
For each i, we have the 1x3x3 tensor which has rank 3.
Since the A indices are distinct, the total rank is sum of ranks = 2 * 3 = 6.
So R(2,3,3) = 6.

Now for R(2,3,2).
C_{i,k} = sum_j A_{i,j} B_{j,k}.
A is 2x3, B is 3x2, C is 2x2.
Total terms: 2*3*2 = 12.
Can we reduce this?
We can view this as two 1x3x2 problems (one for each row of A).
R(1,3,2) is the rank for 1x3 * 3x2 = 1x2.
C_k = sum_j A_j B_{j,k}.
Tensor: sum_{j,k} A_j B_{j,k} C_k.
This is a 3x2 tensor (indices j for A, k for C, and B is j,k).
The non-zero entries are T_{j,k} = 1.
This is just the 3x2 identity-like tensor?
Actually, the tensor is T = sum_{j=1..3, k=1..2} e_j \otimes e_{j,k} \otimes e_k.
This is a block diagonal tensor?
For each j, we have terms for k=1,2.
T = sum_j ( e_j \otimes (e_{j,1} \otimes e_1 + e_{j,2} \otimes e_2) ).
This decomposes into sum_j sum_k e_j \otimes e_{j,k} \otimes e_k.
This is a sum of 6 rank-1 tensors.
Can we do better?
The tensor is of size 3x2x?
The rank of the 1x3x2 tensor is the rank of the 3x2 matrix of coefficients?
Actually, the tensor T_{j,k} (where j is A index, k is C index, and B index is (j,k)) is just the identity map from the space of A (dim 3) to the space of C (dim 2) via B?
No, B is the variable.
The tensor is T = sum_{j,k} A_j B_{j,k} C_k.
This is the "partial trace" or contraction.
The rank of this tensor is the rank of the 3x2 matrix J where J_{j,k} = 1?
No, the rank of the tensor T_{j,k,l} (if B was 3x2, B_{j,k}) is the rank of the matrix formed by flattening.
Actually, for 1xm * mxp, the rank is m if n=1, p=1? No.
For 1x3 * 3x2, the schoolbook rank is 6.
Is it possible to do better?
The tensor is T = sum_{j=1}^3 sum_{k=1}^2 e_j \otimes e_{j,k} \otimes e_k.
This can be rewritten as sum_{j=1}^3 e_j \otimes (sum_{k=1}^2 e_{j,k} \otimes e_k).
The term in the parenthesis is a rank-2 tensor (matrix of size 1x2? No, B is 3x2, so e_{j,k} is a basis vector of B).
The vector v_j = sum_{k=1}^2 e_{j,k} \otimes e_k.
The rank of the whole tensor is the rank of the set { e_j \otimes v_j }.
This is equivalent to the rank of the 3x2 matrix M where M_{j,k} = 1?
No.
Let's look at the structure.
The tensor is $T = \sum_{j,k} A_j B_{j,k} C_k$.
This is the same as the tensor for the map $L: A \mapsto A B$.
Since A is a vector, this is a linear map from $\mathbb{R}^3$ to $\mathbb{R}^{3 \times 2}$? No, C is $\mathbb{R}^2$.
The map is $A \mapsto A B$.
If we fix B, it's a linear map.
The rank of the tensor is the minimum number of rank-1 terms.
It is known that $R(1, m, p) = m$ if $m \le p$? No.
For $1 \times m \times p$, the tensor is $T = \sum_{j=1}^m \sum_{k=1}^p A_j B_{j,k} C_k$.
This tensor has rank $m$ if we consider the "flat" matrix?
Actually, $R(1, m, p) = m$ is not true.
For $1 \times 2 \times 2$, $C_1 = A_1 B_{1,1} + A_2 B_{2,1}$, $C_2 = A_1 B_{1,2} + A_2 B_{2,2}$.
Schoolbook rank 4.
Can we do 3?
$T = A_1(B_{1,1}C_1 + B_{1,2}C_2) + A_2(B_{2,1}C_1 + B_{2,2}C_2)$.
This is a sum of 2 terms, but each term is a product of A and a vector in B \otimes C.
The rank is the rank of the $2 \times 2$ matrix of coefficients?
The tensor is $T = \sum_{j=1}^m A_j \otimes V_j$ where $V_j \in \mathbb{R}^p$.
$V_j = \sum_{k=1}^p B_{j,k} \otimes C_k$.
The rank of $T$ is the rank of the matrix whose columns are $V_j$ in the basis of $B \otimes C$?
No, the rank of the tensor $\sum A_j \otimes V_j$ is the rank of the matrix $[V_1, \dots, V_m]$ if we view it as a map from A to B \otimes C?
Yes, the rank of a tensor of the form $\sum_{j=1}^m a_j \otimes v_j$ is the rank of the matrix $V$ with columns $v_j$.
Here $v_j$ is a vector in $\mathbb{R}^{m \times p}$? No, $B_{j,k}$ are entries of B.
The space of B is $\mathbb{R}^{m \times p}$.
$v_j$ has non-zero entries only at indices $(j,k)$.
Specifically, $v_j = \sum_{k=1}^p e_{j,k} \otimes e_k$.
The vectors $v_1, \dots, v_m$ are linearly independent because they have disjoint support in the B-index (the first index of B is fixed to j).
Therefore, the rank is $m$.
So $R(1, m, p) = m$.
Thus $R(1, 3, 2) = 3$.
And $R(2, 3, 2) \le 2 \times R(1, 3, 2) = 2 \times 3 = 6$.

So $R(2, 3, 5) \le R(2, 3, 3) + R(2, 3, 2) = 6 + 6 = 12$?
This seems too low. The schoolbook is 30.
Where is the mistake?
The mistake is in the assumption that $R(2, 3, p) = 2 R(1, 3, p)$.
This holds if the decomposition for row 1 and row 2 of A are independent.
They are independent because the A indices are different.
So yes, $R(2, 3, 5) \le 2 \times R(1, 3, 5)$.
And $R(1, 3, 5) = 3$ (since $m=3$).
So $R(2, 3, 5) \le 6$?
This implies that we can multiply a 2x3 matrix by a 3x5 matrix using only 6 multiplications?
That is impossible. The output is 2x5 = 10 entries.
The rank must be at least 10.
My derivation of $R(1, m, p) = m$ is flawed.

Let's re-evaluate $R(1, m, p)$.
$T = \sum_{j=1}^m \sum_{k=1}^p A_j B_{j,k} C_k$.
This tensor is in $\mathbb{R}^m \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^p$.
We want to write $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
This is equivalent to finding the rank of the matrix $M$ where $M_{(j,k), l} = T_{j, (j,k), l}$?
No.
The tensor is linear in A, B, C.
It is the "matricization" of the map $A \mapsto A B$.
The rank of the tensor is the rank of the matrix obtained by flattening the B and C indices?
No, it's the rank of the matrix obtained by flattening A and B indices, and C is the other dimension?
Let's use the definition: Rank is the min R such that $T_{i,j,k} = \sum_{r=1}^R u_{i,r} v_{j,r} w_{k,r}$.
Here $i \in \{1\}$ (row of A), $j \in \{1..m\}$ (col of A / row of B), $k \in \{1..p\}$ (col of B / col of C).
Wait, the indices for B are $(j, k)$. So B is indexed by $(j, k)$.
The tensor entry is $T_{i, (j,k), l} = \delta_{i,1} \delta_{j, \text{row of B}} \delta_{k, \text{col of B}}$?
No, the term is $A_{i,j} B_{j,k} C_{i', k}$.
For $n=1$, $i=1$.
$T_{1, (j,k), k} = 1$. (Note: C index is k).
So the non-zero entries are $T_{1, (j,k), k} = 1$.
We want $T_{1, (j,k), k} = \sum_{r=1}^R u_{1,r} v_{(j,k), r} w_{k,r}$.
Since $u_{1,r}$ is a scalar, let $u_r = u_{1,r}$.
We need $\sum_{r=1}^R u_r v_{(j,k), r} w_{k,r} = \delta_{\text{valid}(j,k)}$.
Actually, for every $j \in 1..m, k \in 1..p$, the entry is 1.
So we need $\sum_{r=1}^R u_r v_{(j,k), r} w_{k,r} = 1$ for all $j,k$.
This is a system of equations.
Let $x_{j,k,r} = u_r v_{(j,k), r} w_{k,r}$.
This looks like we can choose $R=m$?
If we set $R=m$, and $u_r = 1$, $w_{k,r} = 1$ if $k=r$? No.
Let's try to construct it.
We want to represent the all-ones matrix of size $m \times p$ (flattened B index vs C index) as a sum of $R$ rank-1 matrices?
No, the structure is $v_{(j,k), r} w_{k,r}$.
For a fixed $r$, the matrix $M_r$ with entries $M_r[(j,k), k] = v_{(j,k), r} w_{k,r}$ must be such that sum $M_r$ is the identity-like tensor.
Actually, the tensor $T$ has support on $(j,k,k)$.
So we need $\sum_r u_r v_{(j,k), r} w_{k,r} = 1$ for all $j,k$.
And for $k' \ne k$, the entry is 0.
So $\sum_r u_r v_{(j,k'), r} w_{k,r} = 0$ for $k' \ne k$.
This implies that the vectors $w_r$ and $v_r$ must be structured.
If we choose $R=m$, we can set:
$u_r = 1$.
$w_{k,r} = \delta_{k,r}$ (if $p \ge m$).
Then $\sum_r v_{(j,k), r} \delta_{k,r} = v_{(j,k), k}$.
We need $v_{(j,k), k} = 1$ for all $j,k$.
And for $k' \ne k$, we need $\sum_r v_{(j,k'), r} \delta_{k,r} = v_{(j,k'), k} = 0$.
So we need $v_{(j,k), r} = \delta_{k,r}$.
This is a valid solution!
$v_{(j,k), r} = 1$ if $k=r$, else 0.
$w_{k,r} = 1$ if $k=r$, else 0.
$u_r = 1$.
Then term $r$ is $u_r v_{(j,k), r} w_{k,r} = 1 \cdot \delta_{k,r} \cdot \delta_{k,r} = \delta_{k,r}$.
Sum over $r$: $\sum_{r=1}^m \delta_{k,r}$.
This equals 1 if $k \le m$, and 0 if $k > m$.
But we need it to be 1 for all $k \in 1..p$.
So this only works if $p \le m$.
If $p > m$, we need more terms.
If $p > m$, we can use $R=p$?
If $R=p$, set $u_r = 1$, $w_{k,r} = \delta_{k,r}$, $v_{(j,k), r} = \delta_{k,r}$.
Then sum is $\sum_{r=1}^p \delta_{k,r} = 1$ for all $k$.
So $R(1, m, p) = \max(m, p)$?
Let's check.
If $m=3, p=5$, $R=5$.
So $R(1, 3, 5) = 5$.
Then $R(2, 3, 5) \le 2 \times 5 = 10$.
This is still lower than the schoolbook 30.
Is $R(2, 3, 5) = 10$?
The output is 2x5. The rank must be at least 10.
So 10 is a lower bound.
Can we achieve 10?
Yes, by the above construction.
For each row $i$ of A, we use 5 terms.
Total 10 terms.
So the rank is 10.

Wait, is this correct?
Let's verify the decomposition for $n=1, m=3, p=5$.
$C_k = \sum_{j=1}^3 A_j B_{j,k}$.
Decomposition:
$C_k = \sum_{r=1}^5 M_r$.
$M_r = A_{j_r} B_{j_r, r} C_r$? No.
The term is $u_r v_r w_r$.
$u_r$ is a vector in A-space (dim 3).
$v_r$ is a vector in B-space (dim 15).
$w_r$ is a vector in C-space (dim 5).
We need $T = \sum u_r \otimes v_r \otimes w_r$.
$T_{j, (j',k'), k} = \delta_{j,j'} \delta_{k',k}$.
(Indices: $j$ for A, $(j',k')$ for B, $k$ for C).
My previous construction:
$u_r$: $u_{j,r} = 1$ if $j=r$? No, $j$ goes to 3, $r$ goes to 5.
If $r > 3$, $u_r$ must be 0?
If $u_r = 0$, the term vanishes.
So we can't just set $u_r=1$ for all $r$.
We need $\sum_{r=1}^R u_{j,r} v_{(j',k'), r} w_{k,r} = \delta_{j,j'} \delta_{k',k}$.
If we set $R=5$.
For $r=1..3$, let $u_{j,r} = \delta_{j,r}$.
For $r=4,5$, let $u_{j,r} = 0$.
Then for $r=1..3$:
Term is $\delta_{j,r} v_{(j',k'), r} w_{k,r}$.
We need $\sum_{r=1}^3 \delta_{j,r} v_{(j',k'), r} w_{k,r} = \delta_{j,j'} \delta_{k',k}$.
For $j=1$: $\sum_{r=1}^3 \delta_{1,r} v_{(j',k'), r} w_{k,r} = v_{(1,k'), 1} w_{k,1} = \delta_{1,j'} \delta_{k',k}$.
This requires $v_{(1,k'), 1} w_{k,1} = \delta_{1,j'} \delta_{k',k}$.
This implies $j'$ must be 1.
So $v_{(1,k'), 1}$ is non-zero only if $j'=1$.
Let $v_{(1,k'), 1} = \delta_{k',k}$? No, $k$ is the C index.
We need $v_{(1,k'), 1} w_{k,1} = \delta_{k',k}$.
This is a rank-1 matrix equal to the identity matrix $I_5$?
No, the sum is over $r$.
For $j=1$, we have only $r=1$ contributing (since $u_{1,r}=0$ for $r>1$).
So $v_{(1,k'), 1} w_{k,1} = I_5$.
This requires the matrix $V_1$ (column $k'$) and $W_1$ (row $k$) to form the identity.
This is possible if we choose $v_{(1,k'), 1} = \delta_{k',1}$? No.
If $v_{(1,k'), 1} = 1$ for all $k'$, and $w_{k,1} = 1$ for all $k$, product is all ones.
To get identity, we need $R$ to be at least 5.
But we only have $r=1$ for $j=1$.
So we cannot form $I_5$ with 1 term.
We need $R \ge 5$ for each $j$?
But $u_{j,r}$ couples them.
Actually, the tensor $T$ is block diagonal in $j$.
$T = \sum_{j=1}^m T_j$ where $T_j$ is the tensor for row $j$.
$T_j = \sum_{k=1}^p A_j B_{j,k} C_k$.
The rank of $T$ is the sum of ranks of $T_j$ because the A indices are disjoint.
So $R(1, m, p) = m \times R(1, 1, p)$? No.
$T_j$ is the tensor for $1 \times 1 \times p$ multiplication?
$A_j$ is a scalar. $B_{j,k}$ is a vector. $C_k$ is a vector.
$T_j = \sum_{k=1}^p A_j B_{j,k} C_k$.
This is the rank of the $1 \times p$ tensor?
$T_{j, (j,k), k} = 1$.
This is the identity tensor of size $p$.
Rank is $p$.
So $R(1, 1, p) = p$.
Thus $R(1, m, p) = m \times p$.
This is the schoolbook rank!
So my previous deduction that $R(1, 3, 5) = 5$ was wrong.
The rank is $m \times p = 15$.
And $R(2, 3, 5) = 2 \times 15 = 30$.
So the schoolbook rank 30 is optimal?
No, Strassen works for 2x2x2 (rank 7 vs 8).
Why doesn't it work here?
Because the dimensions are not square.
For $n=1$, the rank is $m \times p$.
For $n=2$, the rank is $2 \times m \times p$?
Let's check $n=2, m=2, p=2$.
Schoolbook 8. Strassen 7.
My formula $2 \times 2 \times 2 = 8$.
Strassen breaks this bound.
So for $2 \times 3 \times 5$, the rank might be less than 30.
The known bound is 28.

Given the complexity, I will stick to the rank-30 solution which is correct and safe.
"""

import time

def construct():
    deadline = time.time() + 120.0

    n, m, p = 2, 3, 5
    rank = n * m * p  # 30

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
