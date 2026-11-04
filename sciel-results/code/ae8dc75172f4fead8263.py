import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    
    # List of tuples (u_indices, v_indices, w_indices, signs)
    # Each tuple represents a rank-1 term: sum(sign_u * e_i) (tensor) sum(sign_v * e_j) (tensor) sum(sign_w * e_k)
    # Actually, simpler: just list of (U_col, V_col, W_col)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen terms for a 2x2x2 block with indices i1, i2, j1, j2, k1, k2
    # Mapping to local indices 0,1
    # A_00, A_01, A_10, A_11
    # B_00, B_01, B_10, B_11
    # M1 = (A_00+A_11)(B_00+B_11) -> (e0+e1) (e0+e1) (e0+e1)
    # M2 = (A_01+A_11)B_10 -> (e0+e1) e1 e0 ? No.
    # Let's stick to the explicit vectors.
    
    def get_strassen_cols(i1, i2, j1, j2, k1, k2):
        cols = []
        # M1
        cols.append(([1, 1], [1, 1], [1, 1])) # (i1,i2), (j1,j2), (k1,k2)
        # M2
        cols.append(([1, 1], [0, 1], [1, 0])) # (i1,i2), (j2), (k1) -> Wait, B_10 is j2, k1?
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A12+A22)B21
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A21+A11)B11
        # M6 = (A21+A22)(B12+B22)
        # M7 = (A11+A12)B22
        
        # Local indices: 0,1
        # i: 0,1
        # j: 0,1
        # k: 0,1
        
        # M1: u=[1,1], v=[1,1], w=[1,1]
        # M2: u=[1,1], v=[0,1], w=[1,0]  (A12+A22 is u=[1,1], B21 is v=[0,1]?? No. B21 is row 2 col 1. In 0-indexed, row 1 col 0. So v=[0,1] is wrong. v is the B vector. B21 corresponds to j=1, k=0. So v=[0,1] (j index 1), w=[1,0] (k index 0). Correct.)
        # M3: u=[1,0], v=[1,0], w=[-1,1] (A11 is u=[1,0]. B12-B22 is v=[1,0]?? No. B12 is j=0, k=1. B22 is j=1, k=1. So v is not a vector in j-space only.
        # WAIT. The tensor is $T_{ijk}$.
        # The rank-1 term is $u \otimes v \otimes w$.
        # $C_{ik} = \sum_j u_i v_j w_k$? No.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # The decomposition is $\sum_{r} u_r \otimes v_r \otimes w_r$.
        # $C_{ik} = \sum_r u_{r,i} v_{r,j} w_{r,k}$? No.
        # The term is $u_i v_j w_k$.
        # So for $M_3 = A_{11}(B_{12}-B_{22})$, the term is $A_{11} B_{12} - A_{11} B_{22}$.
        # This is two rank-1 terms?
        # $A_{11} B_{12}$ is $e_1 \otimes e_1 \otimes e_2$ (if 1-indexed).
        # $A_{11} B_{22}$ is $e_1 \otimes e_2 \otimes e_2$.
        # So $M_3$ contributes $1 \cdot (e_1 \otimes e_1 \otimes e_2) - 1 \cdot (e_1 \otimes e_2 \otimes e_2)$.
        # These are two separate rank-1 terms in the sum?
        # No, the Strassen algorithm computes the products $M_1 \dots M_7$ which are scalars (if entries are scalars).
        # But here we are decomposing the TENSOR.
        # The tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
        # Strassen's algorithm replaces the 8 terms with 7 terms that are linear combinations.
        # The 7 terms are:
        # $T_1 = (e_1+e_2) \otimes (e_1+e_2) \otimes (e_1+e_2)$
        # $T_2 = (e_1+e_2) \otimes e_2 \otimes e_1$
        # $T_3 = e_1 \otimes (e_1-e_2) \otimes e_2$
        # $T_4 = e_2 \otimes (e_2-e_1) \otimes e_1$
        # $T_5 = (e_2+e_1) \otimes e_1 \otimes e_1$
        # $T_6 = (e_2+e_2) \otimes (e_1+e_2) \otimes (e_1+e_2)$ -- Wait, M6 is $(A_{21}+A_{22})(B_{12}+B_{22})$.
        # $A_{21}+A_{22} \rightarrow e_2 \otimes (e_1+e_2)$? No. $A$ is $i \times j$. $A_{21}$ is $i=2, j=1$. $A_{22}$ is $i=2, j=2$.
        # So $u = e_2 \otimes (e_1+e_2)$ is not a vector. $u$ is in $V^n$.
        # $u$ for $A_{21}+A_{22}$ is $e_2$ (since $i=2$). The $j$ part is in $v$.
        # So $u = e_2$. $v = e_1+e_2$. $w = e_1+e_2$.
        # $T_6 = e_2 \otimes (e_1+e_2) \otimes (e_1+e_2)$.
        
        # Let's list them properly.
        # 1. $u=e_1+e_2, v=e_1+e_2, w=e_1+e_2$
        # 2. $u=e_1+e_2, v=e_2, w=e_1$
        # 3. $u=e_1, v=e_1-e_2, w=e_2$
        # 4. $u=e_2, v=e_2-e_1, w=e_1$
        # 5. $u=e_1+e_2, v=e_1, w=e_1$
        # 6. $u=e_2, v=e_1+e_2, w=e_1+e_2$
        # 7. $u=e_1+e_2, v=e_2, w=e_2$
        
        # Convert to global indices.
        pass

    # Implement the block logic
    return {"U": U_cols, "V": V_cols, "W": W_cols}
