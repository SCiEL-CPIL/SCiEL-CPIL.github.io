import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
        
    # Helper for Strassen 2x2x2
    def add_strassen(a_rows, a_cols, b_rows, b_cols):
        # a_rows: list of 2 global row indices for A
        # a_cols: list of 2 global col indices for A
        # b_rows: list of 2 global row indices for B
        # b_cols: list of 2 global col indices for B
        # A is 2x2, B is 2x2.
        # U indices: a_rows[r] * m + a_cols[c]
        # V indices: b_rows[r] * p + b_cols[c]
        # W indices: b_cols[c] * n + a_rows[r]
        
        # P1 = (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22. W: C11, C22? No, W is for C.
        # The decomposition is sum of rank-1 tensors.
        # Tensor T_{ijk} = sum_c U_{i,c} V_{j,c} W_{k,c}.
        # Here i is (row, col) of A? No.
        # U is (n*m, rank). Index (i, j) of A.
        # V is (m*p, rank). Index (j, k) of B.
        # W is (p*n, rank). Index (k, i) of C.
        
        # Strassen formulas for C = AB.
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We need to add 7 terms.
        
        # Term 1: P1
        # (A11+A22) (B11+B22)
        # U: A11=1, A22=1.
        # V: B11=1, B22=1.
        # W: C11=1, C22=1? No.
        # The product is (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This contributes to C11, C12, C21, C22?
        # No, Strassen works on the 2x2 blocks.
        # Let A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]].
        # C = [[C11, C12], [C21, C22]].
        # P1 = (A11+A22)(B11+B22).
        # This is a scalar product of two vectors? No, A11 etc are scalars in the 2x2x2 case.
        # In the tensor decomposition, we are decomposing the bilinear map.
        # For 2x2x2, the "entries" are scalars.
        # So for P1, we add 1 to U[A11], 1 to U[A22].
        # 1 to V[B11], 1 to V[B22].
        # 1 to W[C11], 1 to W[C22]?
        # Wait, the term is $U_{A11} V_{B11} W_{C11} + U_{A11} V_{B22} W_{C12} + \dots$?
        # No. The rank-1 tensor is $u \otimes v \otimes w$.
        # $T_{ijk} = u_i v_j w_k$.
        # If we want to compute $C_{11} = A_{11}B_{11} + A_{12}B_{21} + \dots$,
        # we need to sum over the terms.
        
        # Let's look at the standard Strassen tensor decomposition.
        # It consists of 7 terms.
        # Term 1: $u=(1,0,0,1)$ on A, $v=(1,0,0,1)$ on B, $w=(1,0,0,1)$ on C?
        # No.
        # $P_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
        # This term contributes to:
        # $C_{11} \leftarrow P_1$
        # $C_{12} \leftarrow 0$? No.
        # $C_{11} = A_{11}B_{11} + A_{12}B_{21}$
        # $C_{12} = A_{11}B_{12} + A_{12}B_{22}$
        # $C_{21} = A_{21}B_{11} + A_{22}B_{21}$
        # $C_{22} = A_{21}B_{12} + A_{22}B_{22}$
        
        # $P_1 = A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}$.
        # So $P_1$ contains parts of $C_{11}, C_{12}, C_{21}, C_{22}$.
        # Specifically:
        # $C_{11}$ gets $A_{11}B_{11}$ (part of P1)
        # $C_{12}$ gets $A_{11}B_{22}$ (part of P1)
        # $C_{21}$ gets $A_{22}B_{11}$ (part of P1)
        # $C_{22}$ gets $A_{22}B_{22}$ (part of P1)
        
        # So for Term 1, we set:
        # U[A11] += 1, U[A22] += 1
        # V[B11] += 1, V[B22] += 1
        # W[C11] += 1, W[C12] += 1, W[C21] += 1, W[C22] += 1?
        # No, the tensor is $u \otimes v \otimes w$.
        # $T_{ijk} = u_i v_j w_k$.
        # If we add this term, it adds $1 \cdot 1 \cdot w_k$ to $T_{i,j,k}$ for all $i \in \{11,22\}, j \in \{11,22\}$.
        # So $T_{11,11,11} += 1$ (A11, B11, C11)
        # $T_{11,11,12} += 1$ (A11, B11, C12) -> This is wrong. A11*B11 is in C11, not C12.
        
        # Ah, the indices in U, V, W correspond to the specific entries.
        # $U_{i,j}$ is the coefficient for $A_{i,j}$.
        # $V_{j,k}$ is the coefficient for $B_{j,k}$.
        # $W_{k,i}$ is the coefficient for $C_{k,i}$.
        
        # The term $P_1$ is not a single rank-1 tensor in the global sense if we mix indices.
        # Strassen's 7 terms are rank-1 in the tensor space of the 2x2x2 multiplication.
        # Let's verify the rank-1 structure.
        # Term 1: $u = (1, 0, 0, 1)$ (A11, A12, A21, A22).
        # $v = (1, 0, 0, 1)$ (B11, B12, B21, B22).
        # $w = (1, 0, 0, 1)$ (C11, C12, C21, C22).
        # Product: $A_{11}B_{11}C_{11} + A_{11}B_{22}C_{12} + A_{22}B_{11}C_{21} + A_{22}B_{22}C_{22}$.
        # But we want $A_{11}B_{11}$ to go to $C_{11}$.
        # $A_{11}B_{22}$ goes to $C_{12}$? Yes, $A_{11}B_{22}$ is in $C_{12}$?
        # $C_{12} = A_{11}B_{12} + A_{12}B_{22}$.
        # So $A_{11}B_{22}$ is NOT in $C_{12}$. It's not in any C?
        # Wait. $A_{11}B_{22}$ is the product of top-left of A and bottom-right of B.
        # In standard multiplication, $A_{11}B_{22}$ does not appear directly in C.
        # $C_{11} = A_{11}B_{11} + A_{12}B_{21}$.
        # $C_{12} = A_{11}B_{12} + A_{12}B_{22}$.
        # $C_{21} = A_{21}B_{11} + A_{22}B_{21}$.
        # $C_{22} = A_{21}B_{12} + A_{22}B_{22}$.
        
        # So $A_{11}B_{22}$ is a "cross term" that must cancel out or be combined.
        # In Strassen, $P_1 = (A_{11}+A_{22})(B_{11}+B_{22}) = A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}$.
        # This term is added to the sum.
        # Other terms will cancel the unwanted parts.
        # So the rank-1 tensor for P1 is indeed:
        # $u = [1, 0, 0, 1]$
        # $v = [1, 0, 0, 1]$
        # $w = [1, 1, 1, 1]$?
        # If $w=[1,1,1,1]$, then $A_{11}B_{11}$ contributes to $C_{11}, C_{12}, C_{21}, C_{22}$.
        # This is correct for the intermediate sum. The final C is obtained by linear combination of P's.
        # But in the tensor decomposition $T = \sum u_c \otimes v_c \otimes w_c$,
        # $T_{ijk}$ is the coefficient of $A_i B_j$ in $C_k$.
        # For 2x2x2, $T_{A_{11}, B_{11}, C_{11}} = 1$.
        # $T_{A_{11}, B_{12}, C_{11}} = 0$.
        # $T_{A_{11}, B_{11}, C_{12}} = 0$.
        
        # So we must check the $w$ vector.
        # For P1, we want $A_{11}B_{11} \to C_{11}$, $A_{11}B_{22} \to C_{12}$? No.
        # The term $P_1$ is just a scalar. It is added to $C_{11}, C_{12}, C_{21}, C_{22}$ with coefficients from the final formula.
        # But in the tensor decomposition, each rank-1 term corresponds to a specific mapping.
        # Actually, the 7 terms of Strassen are rank-1 tensors in the space of $2 \times 2 \times 2$ matrices?
        # No, the rank of the multiplication tensor is 7.
        # The 7 vectors $u_c, v_c, w_c$ define the decomposition.
        # For $c=1$ (P1):
        # $u_1 = (1, 0, 0, 1)$ (A11, A22)
        # $v_1 = (1, 0, 0, 1)$ (B11, B22)
        # $w_1 = (1, 0, 0, 1)$ (C11, C22)?
        # If $w_1 = (1, 0, 0, 1)$, then $A_{11}B_{11}$ contributes to $C_{11}$ and $C_{22}$.
        # $A_{11}B_{22}$ contributes to $C_{11}$ and $C_{22}$.
        # This matches the "intermediate" values.
        
        # Let's check the final formulas again.
        # $C_{11} = P_1 + P_4 - P_5 + P_7$
        # $C_{12} = P_3 + P_5$
        # $C_{21} = P_2 + P_4$
        # $C_{22} = P_1 - P_2 + P_3 + P_6$
        
        # So for Term 1 (P1):
        # $w_1$ should have 1 at $C_{11}$ and 1 at $C_{22}$.
        # $u_1$ has 1 at $A_{11}, A_{22}$.
        # $v_1$ has 1 at $B_{11}, B_{22}$.
        # This works.
        
        # Term 2 (P2): $(A_{21}+A_{22})B_{11}$
        # $u_2 = (0, 0, 1, 1)$
        # $v_2 = (1, 0, 0, 0)$
        # $w_2 = (0, 0, 1, -1)$?
        # $C_{21} += P_2$, $C_{22} -= P_2$.
        # So $w_2$ has 1 at $C_{21}$, -1 at $C_{22}$.
        
        # And so on.
        
        pass

    # We will implement the full recursion in the code below.
    return {"U": U, "V": "V", "W": W}
# EVOLVE-BLOCK-END
