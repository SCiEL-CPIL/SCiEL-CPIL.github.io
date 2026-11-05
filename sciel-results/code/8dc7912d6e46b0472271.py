import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Helper to add a term to the global matrices
    # U is n*m x rank, V is m*p x rank, W is p*n x rank
    # We will build lists of terms first, then assemble.
    # Term: (u_idx, v_idx, w_idx, val)
    # But U, V, W are matrices.
    # Let's just collect the (U_row, V_row, W_row, value) for each term?
    # No, each term c contributes U[:, c], V[:, c], W[:, c].
    # So we just need to append columns to U, V, W.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_vec, V_vec, W_vec):
        """
        U_vec: list of length n*m
        V_vec: list of length m*p
        W_vec: list of length p*n
        """
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)

    def make_zeros(size):
        return [0] * size

    def strassen_2x2x2(offset_i, offset_j, offset_k):
        """
        Add Strassen decomposition for 2x2x2 block at given offsets.
        """
        # Strassen terms for 2x2x2
        # U, V, W are 4x1, 4x1, 4x1 for each term? No, 2x2 matrices flattened.
        # U index: i*2 + j
        # V index: j*2 + k
        # W index: k*2 + i
        
        # Term 1: P1 = (A00+A11)(B00+B11)
        # U: [1, 0, 0, 1] (A00, A01, A10, A11) -> A00+A11 is [1,0,0,1]
        # V: [1, 1, 1, 1] (B00, B01, B10, B11) -> B00+B11? No.
        # V needs to compute B00+B11?
        # Wait, Strassen P1 = (A00+A11)(B00+B11).
        # This is not a single product of linear forms in the standard tensor decomposition sense unless we expand?
        # No, the tensor decomposition is $\sum U_{ij} V_{jk} W_{ki}$.
        # For P1, we need $U$ to pick $A_{00}+A_{11}$, $V$ to pick $B_{00}+B_{11}$?
        # But $V$ is multiplied by $U$ in the sum over $j$.
        # $C_{ik} = \sum_j U_{ij} V_{jk} W_{ki}$.
        # If we set $U_{00}=1, U_{11}=1$ and $V_{00}=1, V_{11}=1$?
        # Then $\sum_j U_{0j} V_{j0} = U_{00}V_{00} + U_{01}V_{10} = 1*1 + 0*0 = 1$.
        # This doesn't give $A_{00}+A_{11}$.
        
        # The standard Strassen decomposition is:
        # M1 = (A00+A11)(B00+B11)
        # This corresponds to a term where:
        # U picks A00 and A11.
        # V picks B00 and B11.
        # But the contraction is over j.
        # So we need $U_{ij}$ and $V_{jk}$ such that their product sums to the desired linear forms.
        
        # Let's use the known rank-7 decomposition vectors.
        # Reference: 
        # U1 = [1, 0, 0, 1] (A00+A11) -- Wait, U is 2x2.
        # Actually, the vectors are:
        # Term 1:
        # U1 = [[1, 0], [0, 1]] -> Flat: [1, 0, 0, 1]
        # V1 = [[1, 1], [1, 1]] -> Flat: [1, 1, 1, 1]
        # W1 = [[1, 0], [0, 1]] -> Flat: [1, 0, 0, 1]
        # Check: C00 = U1[0,0]V1[0,0]W1[0,0] + U1[0,1]V1[1,0]W1[0,1] ...
        # This is getting confusing. Let's use the explicit 7 terms from literature.
        
        # Terms (U_flat, V_flat, W_flat)
        # 1. ( [1,0,0,1], [1,1,1,1], [1,0,0,1] ) -> P1
        # 2. ( [0,0,1,1], [1,0,0,0], [0,1,0,0] ) -> P2?
        # Let's verify P2 = (A10+A11)B00.
        # U2 = [0,0,1,1] (A10+A11)
        # V2 = [1,0,0,0] (B00)
        # W2 = [0,1,0,0] (Contributes to C10? C10 is i=1, k=0. W[0,1]=1)
        # C10 = sum_j U[1,j] V[j,0] W[0,1]
        # U[1,0]=1, U[1,1]=1. V[0,0]=1, V[1,0]=0.
        # Sum = 1*1*1 + 1*0*1 = 1. Correct.
        
        # Let's generate the 7 terms properly.
        
        terms = [
            # P1: (A00+A11)(B00+B11) -> C00, C11
            # U: A00+A11 -> [1,0,0,1]
            # V: B00+B11 -> [1,0,0,1]? No, V[j,k].
            # If V = [1,1,1,1], then V[0,0]=1, V[1,0]=1, etc.
            # We need V such that it picks B00 for j=0 and B11 for j=1?
            # No, Strassen P1 is a single product.
            # The decomposition is:
            # C = P1 * W1 + ...
            
            # Let's use the standard basis.
            # 1. U=[1,0,0,1], V=[1,1,1,1], W=[1,0,0,1]
            # 2. U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,0]
            # 3. U=[1,0,0,0], V=[0,1,1,-1], W=[1,0,0,0]
            # 4. U=[0,0,0,1], V=[1,-1,0,0], W=[0,1,-1,0]
            # 5. U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,1]
            # 6. U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,-1]
            # 7. U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
        ]
        
        # Verify Term 1:
        # U=[1,0,0,1], V=[1,1,1,1], W=[1,0,0,1]
        # C00 = U00*V00*W00 + U01*V10*W01 = 1*1*1 + 0*1*0 = 1.
        # C11 = U10*V01*W10 + U11*V11*W11 = 0*1*0 + 1*1*1 = 1.
        # C01 = U00*V01*W10 + U01*V11*W11 = 1*1*0 + 0*1*1 = 0.
        # C10 = U10*V00*W00 + U11*V10*W01 = 0*1*1 + 1*1*0 = 0.
        # Correct.
        
        for U_flat, V_flat, W_flat in terms:
            U_vec = make_zeros(n * m)
            V_vec = make_zeros(m * p)
            W_vec = make_zeros(p * n)
            
            for local_i in range(2):
                for local_j in range(2):
                    global_i = offset_i + local_i
                    global_j = offset_j + local_j
                    U_vec[global_i * m + global_j] = U_flat[local_i * 2 + local_j]
                    
            for local_j in range(2):
                for local_k in range(2):
                    global_j = offset_j + local_j
                    global_k = offset_k + local_k
                    V_vec[global_j * p + global_k] = V_flat[local_j * 2 + local_k]
                    
            for local_k in range(2):
                for local_i in range(2):
                    global_k = offset_k + local_k
                    global_i = offset_i + local_i
                    W_vec[global_k * n + global_i] = W_flat[local_k * 2 + local_i]
                    
            add_term(U_vec, V_vec, W_vec)

    def schoolbook_block(n_b, m_b, p_b, offset_i, offset_j, offset_k):
        """
        Add schoolbook decomposition for n_b x m_b x p_b block.
        """
        for local_i in range(n_b):
            for local_j in range(m_b):
                for local_k in range(p_b):
                    U_vec = make_zeros(n * m)
                    V_vec = make_zeros(m * p)
                    W_vec = make_zeros(p * n)
                    
                    global_i = offset_i + local_i
                    global_j = offset_j + local_j
                    global_k = offset_k + local_k
                    
                    U_vec[global_i * m + global_j] = 1
                    V_vec[global_j * p + global_k] = 1
                    W_vec[global_k * n + global_i] = 1
                    
                    add_term(U_vec, V_vec, W_vec)

    # Construct the blocks
    
    # Block 1: i[0,1], j[0,1], k[0,2]
    # k[0,1] Strassen
    strassen_2x2x2(0, 0, 0)
    # k[2,2] Schoolbook 2x2x1
    schoolbook_block(2, 2, 1, 0, 0, 2)
    
    # Block 2: i[0,1], j[2,3], k[0,2]
    # k[0,1] Strassen
    strassen_2x2x2(0, 2, 0)
    # k[2,2] Schoolbook 2x2x1
    schoolbook_block(2, 2, 1, 0, 2, 2)
    
    # Block 3: i[0,1], j[0,3], k[3,5]
    # j[0,1], k[3,4] Strassen
    strassen_2x2x2(0, 0, 3)
    # j[0,1], k[5,5] Schoolbook
    schoolbook_block(2, 2, 1, 0, 0, 5)
    
    # j[2,3], k[3,4] Strassen
    strassen_2x2x2(0, 2, 3)
    # j[2,3], k[5,5] Schoolbook
    schoolbook_block(2, 2, 1, 0, 2, 5)
    
    # Block 4: i[2,2], j[0,3], k[0,5]
    # Schoolbook 1x4x6
    schoolbook_block(1, 4, 6, 2, 0, 0)
    
    # Assemble matrices
    rank = len(U_cols)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        U[:, c] = U_cols[c]
        V[:, c] = V_cols[c]
        W[:, c] = W_cols[c]
        
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
