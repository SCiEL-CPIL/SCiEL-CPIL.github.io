import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(a_idx, b_idx, c_idx, u_val, v_val, w_val):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(n * p)
        
        u[a_idx] = u_val
        v[b_idx] = v_val
        w[c_idx] = w_val
        
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def add_strassen(i_off, j_off, k_off):
        # Adds Strassen decomposition for 2x2x2 block
        # A is 2x2, B is 2x2, C is 2x2
        # Indices in local block: i in 0..1, j in 0..1, k in 0..1
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add 7 terms.
        # Each term is a product of a linear combo of A and a linear combo of B.
        # U, V, W matrices will have columns corresponding to these 7 products.
        # However, the standard tensor decomposition format requires rank-1 tensors.
        # A rank-1 tensor is u \otimes v \otimes w.
        # Here, the "multiplication" is the product of the linear forms.
        # So we add 7 columns to U, V, W.
        
        # M1
        # U: A00 + A11. V: B00 + B11. W: C00 + C11 (with coeffs)
        # Wait, the decomposition is T = sum R_k (U_k \otimes V_k \otimes W_k)
        # For Strassen, we express C as linear combos of M_k.
        # C_ij = sum_k coeff_ij_k M_k
        # M_k = (U_k \cdot A) (V_k \cdot B)
        # So the tensor is sum_k (U_k \otimes V_k \otimes W_k) where W_k is the vector of coeffs for C.
        # Actually, the tensor T_{ijk} = sum_k U_{ik} V_{jk} W_{ki} ... no.
        # T_{ijk} = sum_k U_{ik} V_{jk} W_{kj} is not right.
        # T_{ijk} is the coefficient of A_i B_j in C_k.
        # In Strassen, C_k = sum_l M_l * coeff_{kl}.
        # M_l = (sum_i A_i u_{li}) (sum_j B_j v_{lj}).
        # So T_{ijk} = sum_l u_{li} v_{lj} w_{kl} where w_{kl} is coeff of M_l in C_k.
        # So we add 7 terms.
        # Term l:
        # U vector has non-zeros at A indices involved in M_l.
        # V vector has non-zeros at B indices involved in M_l.
        # W vector has non-zeros at C indices involved in M_l, with values from coeff matrix.
        
        # Let's define the 7 forms.
        # Indices: A00=0, A01=1, A10=2, A11=3 (local)
        # B00=0, B01=1, B10=2, B11=3 (local)
        # C00=0, C01=1, C10=2, C11=3 (local)
        
        # M1: A00+A11, B00+B11
        # M2: A10+A11, B00
        # M3: A00, B01-B11
        # M4: A11, B10-B00
        # M5: A00+A01, B11
        # M6: A10-A00, B00+B01
        # M7: A01-A11, B10+B11
        
        # Coeffs for C:
        # C00: M1(1) + M4(1) - M5(1) + M7(1)
        # C01: M3(1) + M5(1)
        # C10: M2(1) + M4(1)
        # C11: M1(1) - M2(1) + M3(1) + M6(1)
        
        # We will add 7 columns.
        
        # Helper to add one Strassen term
        def add_strassen_term(u_local, v_local, w_local):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(n * p)
            
            for li, val in u_local:
                u[(i_off + li[0]) * m + (j_off + li[1])] = val
            for lj, val in v_local:
                v[(j_off + lj[0]) * p + (k_off + lj[1])] = val
            for lk, val in w_local:
                w[(k_off + lk[0]) * n + (i_off + lk[1])] = val
                
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
            
        # Term 1: M1
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        add_strassen_term(
            [((0,0), 1.0), ((1,1), 1.0)],
            [((0,0), 1.0), ((1,1), 1.0)],
            [((0,0), 1.0), ((1,1), 1.0)]
        )
        
        # Term 2: M2
        # U: A10(1), A11(1)
        # V: B00(1)
        # W: C10(1), C11(-1)
        add_strassen_term(
            [((1,0), 1.0), ((1,1), 1.0)],
            [((0,0), 1.0)],
            [((1,0), 1.0), ((1,1), -1.0)]
        )
        
        # Term 3: M3
        # U: A00(1)
        # V: B01(1), B11(-1)
        # W: C01(1), C11(1)
        add_strassen_term(
            [((0,0), 1.0)],
            [((0,1), 1.0), ((1,1), -1.0)],
            [((0,1), 1.0), ((1,1), 1.0)]
        )
        
        # Term 4: M4
        # U: A11(1)
        # V: B10(1), B00(-1)
        # W: C00(1), C10(1)
        add_strassen_term(
            [((1,1), 1.0)],
            [((1,0), 1.0), ((0,0), -1.0)],
            [((0,0), 1.0), ((1,0), 1.0)]
        )
        
        # Term 5: M5
        # U: A00(1), A01(1)
        # V: B11(1)
        # W: C00(-1), C01(1)
        add_strassen_term(
            [((0,0), 1.0), ((0,1), 1.0)],
            [((1,1), 1.0)],
            [((0,0), -1.0), ((0,1), 1.0)]
        )
        
        # Term 6: M6
        # U: A10(1), A00(-1)
        # V: B00(1), B01(1)
        # W: C11(1)
        add_strassen_term(
            [((1,0), 1.0), ((0,0), -1.0)],
            [((0,0), 1.0), ((0,1), 1.0)],
            [((1,1), 1.0)]
        )
        
        # Term 7: M7
        # U: A01(1), A11(-1)
        # V: B10(1), B11(1)
        # W: C00(1)
        add_strassen_term(
            [((0,1), 1.0), ((1,1), -1.0)],
            [((1,0), 1.0), ((1,1), 1.0)],
            [((0,0), 1.0)]
        )

    def add_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    a_idx = (i_off + i) * m + (j_off + j)
                    b_idx = (j_off + j) * p + (k_off + k)
                    c_idx = (k_off + k) * n + (i_off + i)
                    
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(n * p)
                    
                    u[a_idx] = 1.0
                    v[b_idx] = 1.0
                    w[c_idx] = 1.0
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    # Split n=4 into 2+2
    # Split m=5 into 2+3
    # Split p=5 into 2+3
    
    # Top half (i=0,1)
    # Block 1: 2x2x2 (Strassen)
    add_strassen(0, 0, 0)
    
    # Block 2: 2x2x3 (Schoolbook)
    add_schoolbook(2, 2, 3, 0, 0, 2)
    
    # Block 3: 2x3x2 (Schoolbook)
    add_schoolbook(2, 3, 2, 0, 2, 0)
    
    # Block 4: 2x3x3 (Schoolbook)
    add_schoolbook(2, 3, 3, 0, 2, 2)
    
    # Bottom half (i=2,3)
    # Block 5: 2x2x2 (Strassen)
    add_strassen(2, 0, 0)
    
    # Block 6: 2x2x3 (Schoolbook)
    add_schoolbook(2, 2, 3, 2, 0, 2)
    
    # Block 7: 2x3x2 (Schoolbook)
    add_schoolbook(2, 3, 2, 2, 2, 0)
    
    # Block 8: 2x3x3 (Schoolbook)
    add_schoolbook(2, 3, 3, 2, 2, 2)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
