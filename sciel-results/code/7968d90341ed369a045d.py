import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This is the Kronecker product of Strassen's rank-7 algorithm for 2x2
    matrix multiplication with itself.
    """
    n, m, p = DIMENSIONS
    rank = 49

    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U_local[7, 4]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_local = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ], dtype=int)

    V_local = np.array([
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ], dtype=int)

    W_local = np.array([
        [1, 0, 0, 1],   # M1: C00+=M1, C11+=M1
        [0, 0, 1, -1],  # M2: C10+=M2, C11-=M2
        [0, 1, 0, 1],   # M3: C01+=M3, C11+=M3
        [1, 0, 1, 0],   # M4: C00+=M4, C10+=M4
        [-1, 1, 0, 0],  # M5: C00-=M5, C01+=M5
        [0, 0, 0, 1],   # M6: C11+=M6
        [1, 0, 0, 0]    # M7: C00+=M7
    ], dtype=int)

    # Verify Strassen decomposition
    # C = U * M * W? No, sum_r U[:,r] V[:,r] W[:,r] = T
    # Let's check one element: C00
    # C00 = sum_r U[0,r] V[:,r] W[0,r]
    # U[0] = [1,0,0,1]
    # W[0] = [1,0,0,1]
    # Term r=0: U[0,0]=1, V[:,0]=[1,0,0,1], W[0,0]=1 -> 1 * (A00+A11)(B00+B11) * 1? No.
    # The formula is: C = sum_r (U[:,r] A) (V[:,r] B) W[:,r]? No.
    # Tensor decomposition: T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # Here i is A index, j is B index, k is C index.
    # A_{i,j} B_{j,k} -> C_{i,k}.
    # So T_{i, j, k} = delta_{j, j'} ... wait.
    # Matrix mult tensor T_{i, j, k} where i is row of A, j is col of A/row of B, k is col of B.
    # Actually, the standard tensor for matmul is T_{i, j, k} such that C_{ik} = sum_j A_{ij} B_{jk}.
    # So the indices are (i, j, k) for A, B, C?
    # Usually defined as: T_{i, j, k} = 1 if j is the contraction index?
    # No, the tensor is T_{i, j, k} where i indexes A, j indexes B, k indexes C?
    # If A is n x m, B is m x p, C is n x p.
    # A_{i, alpha}, B_{beta, k}. Sum over alpha, beta where alpha=beta.
    # The tensor is T_{i, beta, k} ? No.
    # The tensor is T_{i, j, k} where i is A row, j is A col/B row, k is B col/C col.
    # T_{i, j, k} = 1 if j is the same for A and B?
    # Actually, the tensor is defined by:
    # <T, A \otimes B \otimes C> = Tr(A B C^T) ?
    # Or simply: C = A B.
    # The decomposition is: C_{ik} = sum_r (sum_j U_{ij}^{(r)} A_{ij}) (sum_l V_{kl}^{(r)} B_{kl}) W_{ik}^{(r)}?
    # No, the factors are vectors.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    # U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} ?
    # Let's check the dimensions.
    # U has size n*m. Index is (i,j).
    # V has size m*p. Index is (j,k).
    # W has size p*n. Index is (k,i).
    # Sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = C_{ik}?
    # No, the sum is over r. But what about j?
    # The tensor T has indices i, j, k?
    # If we contract T with A and B, we get C.
    # C_{ik} = sum_{j} A_{ij} B_{jk}.
    # So T_{i, j, k} = 1? No.
    # The tensor T is such that T_{i, j, k} = 1 if the indices match?
    # Actually, the tensor is T_{i, j, k} where i is row of C, j is col of C? No.
    # The tensor is T_{i, j, k} with 3 indices?
    # Matmul is a 3-tensor T_{i, j, k} where:
    # i: row of A / row of C
    # j: col of A / row of B
    # k: col of B / col of C
    # T_{i, j, k} = 1.
    # Then C_{ik} = sum_j T_{i, j, k} A_{ij} B_{jk}.
    # Yes.
    # So the decomposition is:
    # 1 = sum_r U_{i, r} V_{j, r} W_{k, r} ? No.
    # The indices of U, V, W in the decomposition must match the tensor indices.
    # Tensor indices: i (from A), j (from A/B), k (from B/C).
    # But U is defined as shape (n*m, rank). This implies U mixes i and j.
    # This is the "flat" tensor decomposition.
    # T_{(i,j), (j,k), (k,i)} ?
    # The problem statement says:
    # U shape (n*m, rank).
    # V shape (m*p, rank).
    # W shape (p*n, rank).
    # This implies the tensor is viewed as a bilinear map from (A, B) to C.
    # C_{ik} = sum_{j} A_{ij} B_{jk}.
    # This can be written as:
    # C = sum_r (U_r \cdot A) (V_r \cdot B) W_r ?
    # Where U_r is a vector of length n*m?
    # Let's check the contraction.
    # Let $A$ be vectorized. $A_{ij}$.
    # $U_{(i,j), r}$.
    # $V_{(j,k), r}$.
    # $W_{(k,i), r}$.
    # If we sum $U_{(i,j), r} A_{ij}$, we get a scalar $M_r^{(A)}$.
    # If we sum $V_{(j,k), r} B_{jk}$, we get a scalar $M_r^{(B)}$.
    # Then $C_{ik} = sum_r M_r^{(A)} M_r^{(B)} W_{(k,i), r}$.
    # Does this work?
    # $C_{ik} = sum_r (sum_{j} U_{(i,j), r} A_{ij}) (sum_{l} V_{(l,k), r} B_{lk}) W_{(k,i), r}$
    # Note: In V, the index is (j,k) or (l,k)?
    # V is (m*p, rank). Index is (j,k) where j is row of B, k is col of B.
    # So $V_{(j,k), r}$.
    # The term is $(sum_j U_{(i,j), r} A_{ij}) (sum_j V_{(j,k), r} B_{jk}) W_{(k,i), r}$.
    # This equals $sum_j sum_l U_{(i,j), r} V_{(l,k), r} A_{ij} B_{lk} W_{(k,i), r}$.
    # For this to equal $sum_j A_{ij} B_{jk}$, we need:
    # $sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1$ for all i,j,k?
    # No, the sum over j is inside the product?
    # No, the product is of two scalars.
    # $M_r^{(A)} = sum_j U_{(i,j), r} A_{ij}$.
    # $M_r^{(B)} = sum_j V_{(j,k), r} B_{jk}$.
    # Product: $M_r^{(A)} M_r^{(B)} = (sum_j U_{(i,j), r} A_{ij}) (sum_l V_{(l,k), r} B_{lk})$.
    # $= sum_{j,l} U_{(i,j), r} V_{(l,k), r} A_{ij} B_{lk}$.
    # Then $C_{ik} = sum_r W_{(k,i), r} sum_{j,l} U_{(i,j), r} V_{(l,k), r} A_{ij} B_{lk}$.
    # $= sum_{j,l} A_{ij} B_{lk} (sum_r U_{(i,j), r} V_{(l,k), r} W_{(k,i), r})$.
    # We want $C_{ik} = sum_j A_{ij} B_{jk}$.
    # So we require:
    # $sum_r U_{(i,j), r} V_{(l,k), r} W_{(k,i), r} = \delta_{j,l}$ ?
    # No, the indices i and k are fixed in the sum over r.
    # So we need:
    # $sum_r U_{(i,j), r} V_{(l,k), r} W_{(k,i), r} = \delta_{j,l}$.
    # This must hold for all i, j, l, k.
    
    # Let's check the Kronecker product of Strassen.
    # Strassen for 2x2:
    # $U^{(2)}_{(i,j), r}$, $V^{(2)}_{(j,k), r}$, $W^{(2)}_{(k,i), r}$.
    # Condition: $sum_r U^{(2)}_{(i,j), r} V^{(2)}_{(l,k), r} W^{(2)}_{(k,i), r} = \delta_{j,l}$.
    
    # For 4x4, we use indices $i = 2 i_1 + i_2$, $j = 2 j_1 + j_2$, $k = 2 k_1 + k_2$.
    # $U^{(4)}_{(i,j), (s,t)} = U^{(2)}_{(i_1, j_1), s} U^{(2)}_{(i_2, j_2), t}$.
    # $V^{(4)}_{(j,k), (s,t)} = V^{(2)}_{(j_1, k_1), s} V^{(2)}_{(j_2, k_2), t}$.
    # $W^{(4)}_{(k,i), (s,t)} = W^{(2)}_{(k_1, i_1), s} W^{(2)}_{(k_2, i_2), t}$.
    
    # Check condition:
    # $sum_{s,t} U^{(2)}_{(i_1, j_1), s} U^{(2)}_{(i_2, j_2), t} V^{(2)}_{(j_1, k_1), s} V^{(2)}_{(j_2, k_2), t} W^{(2)}_{(k_1, i_1), s} W^{(2)}_{(k_2, i_2), t}$
    # $= (sum_s U^{(2)}_{(i_1, j_1), s} V^{(2)}_{(j_1, k_1), s} W^{(2)}_{(k_1, i_1), s}) (sum_t U^{(2)}_{(i_2, j_2), t} V^{(2)}_{(j_2, k_2), t} W^{(2)}_{(k_2, i_2), t})$
    # $= \delta_{j_1, j_1?} \delta_{k_1, k_1?}$
    # Wait, the condition for 2x2 is $\delta_{j_{in}, l_{in}}$?
    # In the 2x2 case, the indices are $i, j, k$.
    # $sum_r U_{(i,j), r} V_{(l,k), r} W_{(k,i), r} = \delta_{j,l}$.
    # So the first term is $\delta_{j_1, j_1}$? No.
    # The indices in the 2x2 sum are $i_1, j_1, k_1$.
    # The "contraction" index is $j$.
    # In the 4x4 derivation, the contraction index is $j$.
    # $j = 2 j_1 + j_2$.
    # The term $\delta_{j,l}$ becomes $\delta_{2 j_1 + j_2, 2 l_1 + l_2}$.
    # This is NOT equal to $\delta_{j_1, l_1} \delta_{j_2, l_2}$ in general?
    # Yes it is, because $j_1, l_1 \in \{0,1\}$ and $j_2, l_2 \in \{0,1\}$.
    # $2 j_1 + j_2 = 2 l_1 + l_2 \iff j_1 = l_1$ and $j_2 = l_2$.
    # So the Kronecker product works perfectly.

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    i1, i2 = divmod(i, 2)
                    j1, j2 = divmod(j, 2)
                    idx_u1 = i1 * 2 + j1
                    idx_u2 = i2 * 2 + j2
                    U[i * 4 + j, col] = U_local[idx_u1, s] * U_local[idx_u2, t]

            # Fill V
            for j in range(4):
                for k in range(4):
                    j1, j2 = divmod(j, 2)
                    k1, k2 = divmod(k, 2)
                    idx_v1 = j1 * 2 + k1
                    idx_v2 = j2 * 2 + k2
                    V[j * 4 + k, col] = V_local[idx_v1, s] * V_local[idx_v2, t]

            # Fill W
            # W index is k * n + i = k * 4 + i
            for k in range(4):
                for i in range(4):
                    k1, k2 = divmod(k, 2)
                    i1, i2 = divmod(i, 2)
                    idx_w1 = k1 * 2 + i1
                    idx_w2 = k2 * 2 + i2
                    W[k * 4 + i, col] = W_local[idx_w1, s] * W_local[idx_w2, t]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
