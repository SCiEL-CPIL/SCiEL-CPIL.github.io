"""Strassen-based decomposition for 4x4 * 4x5 matrix multiplication.
Uses rank 49 for the 4x4 * 4x4 part and rank 16 for the 4x4 * 1 part.
Total rank 65.
"""
import time
import math
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = DIMENSIONS
NM = N * M          # 16
MP = M * P          # 20
PN = P * N          # 20
SCHOOLBOOK_RANK = N * M * P  # 80

DEADLINE = time.time() + 130.0

def _time_left():
    return DEADLINE - time.time()

def _schoolbook():
    """Guaranteed valid decomposition of rank 80 (entries 0/1)."""
    rank = SCHOOLBOOK_RANK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2 * 2x2 multiplication (rank 7)
    and the coefficient matrices u, v, w for the Strassen formulas.
    U, V, W are 4x7 matrices.
    u, v, w are 7x4 matrices (7 terms, 4 entries).
    """
    # Strassen formulas:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 - M3 + M4 - M6
    # C01 = M2 + M6
    # C10 = M1 + M3
    # C11 = M0 - M2 + M1 - M5
    
    # u[k, idx] coeff of A_idx in first factor of M_k
    # v[k, idx] coeff of B_idx in second factor of M_k
    # w[k, idx] coeff of M_k in C_idx
    
    u = np.zeros((7, 4))
    v = np.zeros((7, 4))
    w = np.zeros((7, 4))
    
    # M0
    u[0, 0] = 1; u[0, 3] = 1
    v[0, 0] = 1; v[0, 3] = 1
    # M1
    u[1, 2] = 1; u[1, 3] = 1
    v[1, 0] = 1
    # M2
    u[2, 0] = 1
    v[2, 1] = 1; v[2, 3] = -1
    # M3
    u[3, 3] = 1
    v[3, 2] = 1; v[3, 0] = -1
    # M4
    u[4, 0] = 1; u[4, 1] = 1
    v[4, 3] = 1
    # M5
    u[5, 2] = 1; u[5, 0] = -1
    v[5, 0] = 1; v[5, 1] = 1
    # M6
    u[6, 1] = 1; u[6, 3] = -1
    v[6, 2] = 1; v[6, 3] = 1
    
    # w
    # C00 (idx 0)
    w[0, 0] = 1; w[3, 0] = -1; w[4, 0] = 1; w[6, 0] = -1
    # C01 (idx 1)
    w[2, 1] = 1; w[6, 1] = 1
    # C10 (idx 2)
    w[1, 2] = 1; w[3, 2] = 1
    # C11 (idx 3)
    w[0, 3] = 1; w[2, 3] = -1; w[1, 3] = 1; w[5, 3] = -1
    
    # Now build U, V, W for the base 2x2 case using these u, v, w?
    # No, U, V, W for 2x2 are the "atomic" Strassen decomposition.
    # The u, v, w above are for the "block" level.
    # For the base 2x2 case, the "entries" are scalars.
    # So U_2, V_2, W_2 are exactly the matrices that implement the 7 terms.
    # U_2[i, k] = u[k, i] ?
    # Let's check.
    # Term k is product of (sum u[k,i] A_i) and (sum v[k,j] B_j).
    # This is a rank-1 term in the tensor if we consider A_i, B_j as vectors?
    # No, the tensor is sum_k (sum_i u[k,i] A_i) (sum_j v[k,j] B_j) (sum_l w[k,l] C_l) ?
    # No, the Strassen decomposition is a sum of 7 terms, each of which is a product of 3 linear forms.
    # Term k: L_A(k) * L_B(k) * L_C(k) ?
    # No, it's L_A(k) * L_B(k) = M_k. Then C = sum w[k,l] M_k.
    # So the tensor is sum_k L_A(k) L_B(k) (sum_l w[k,l] C_l) ?
    # No, the tensor decomposition is sum_k (L_A(k) \otimes L_B(k) \otimes L_C(k)).
    # Where L_C(k) is the linear form that extracts the contribution to C?
    # Actually, the term is L_A(k) L_B(k) L_C'(k) where L_C'(k) is such that
    # sum_k L_A(k) L_B(k) L_C'(k) = sum_{i,j,l} A_i B_j C_l (T_{ijl}).
    # In Strassen, we have M_k = L_A(k) L_B(k).
    # And C_l = sum_k w[k,l] M_k.
    # So C_l = sum_k w[k,l] L_A(k) L_B(k).
    # This means the tensor is sum_k (L_A(k) \otimes L_B(k) \otimes (w[k,:] dot C)).
    # So the third vector is w[k, :] (transposed).
    # So U_2[i, k] = u[k, i].
    # V_2[j, k] = v[k, j].
    # W_2[l, k] = w[k, l].
    
    U2 = u.T  # 4x7
    V2 = v.T  # 4x7
    W2 = w.T  # 4x7
    
    return U2, V2, W2, u, v, w

def _build_strassen_4x4_4x4():
    """
    Builds U, V, W for 4x4 * 4x4 multiplication using Strassen.
    Rank 49.
    """
    U2, V2, W2, u, v, w = _get_strassen_2x2()
    
    # U4, V4, W4 are 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for k in range(7):
        for l in range(7):
            idx = k * 7 + l
            for i in range(4):
                for j in range(4):
                    a = i // 2
                    b = j // 2
                    x = i % 2
                    y = j % 2
                    
                    # U4[i, j, idx] = U2[x, y, l] * u[k, a, b]
                    # Note: u is 7x4, so u[k, a*2+b]
                    U4[i * 4 + j, idx] = U2[x * 2 + y, l] * u[k, a * 2 + b]
                    V4[i * 4 + j, idx] = V2[x * 2 + y, l] * v[k, a * 2 + b]
                    # W4 index is (k_out, i_out) ?
                    # W is (P*N, Rank). Here P=4, N=4.
                    # Index is k_out * 4 + i_out.
                    # In the formula: W4[i_out, j_out, idx] = W2[x, y, l] * w[k, a, b]
                    # Wait, W2 index is (l, k_base) where k_base is the output entry index 0..3.
                    # w[k, a, b] is the coeff of M_k in C_{ab}.
                    # So W4[(a*2+x), (b*2+y), idx] = W2[x*2+y, l] * w[k, a*2+b]
                    W4[(a * 2 + x) * 4 + (b * 2 + y), idx] = W2[x * 2 + y, l] * w[k, a * 2 + b]
                    
    return U4, V4, W4

def construct():
    """Return the rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    t0 = time.time()
    fallback = _schoolbook()
    
    if _time_left() < 10:
        return fallback
        
    try:
        U4, V4, W4 = _build_strassen_4x4_4x4()
        
        # Add the 4x4 * 1 part (column 4 of B)
        # Rank 16
        # U5: 16x16, V5: 20x16, W5: 20x16
        # We append to U4, V4, W4
        
        U_total = np.zeros((16, 49 + 16))
        V_total = np.zeros((20, 49 + 16))
        W_total = np.zeros((20, 49 + 16))
        
        # Copy 4x4 part
        # U4 is 16x49
        # V4 is 16x49 -> need to embed in 20x49 (cols 0..3 of B)
        # W4 is 16x49 -> need to embed in 20x49 (cols 0..3 of C)
        
        U_total[:, 0:49] = U4
        
        # V4: B is 4x4. Indices (j, k) where j in 0..3, k in 0..3.
        # V_total index (j*5 + k, rank)
        for j in range(4):
            for k in range(4):
                V_total[j * 5 + k, 0:49] = V4[j * 4 + k, 0:49]
                
        # W4: C is 4x4. Indices (k, i) where k in 0..3, i in 0..3.
        # W_total index (k*4 + i, rank)
        for k in range(4):
            for i in range(4):
                W_total[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
                
        # Add 4x4 * 1 part
        # C_i4 = sum_j A_ij B_j4
        # Terms for j=0..3, i=0..3
        offset = 49
        for i in range(4):
            for j in range(4):
                c = offset + i * 4 + j
                U_total[i * 4 + j, c] = 1
                V_total[j * 5 + 4, c] = 1
                W_total[4 * 4 + i, c] = 1 # k=4, i=i -> 16+i
                
        # Verify
        # Check residual
        Ua = U_total
        Va = V_total
        Wa = W_total
        
        # Residual check
        U3 = Ua.reshape(4, 4, -1)
        V3 = Va.reshape(4, 5, -1)
        W3 = Wa.reshape(5, 4, -1)
        
        B = np.einsum('ijr,jkr->ikr', U3, V3)
        C = np.einsum('ikr,kir->ik', B, W3)
        
        # C[i, k] should be sum_j 1 = 4 for k<4?
        # Wait, C[i, k] = sum_j A_ij B_jk.
        # The tensor is T[i, j, k] = 1.
        # So sum_r U[i, j, r] V[j, k, r] W[k, i, r] should be 1 for each (i, j, k)?
        # No, the decomposition is sum_r U[i, j, r] V[j, k, r] W[k, i, r] = T[i, j, k] = 1.
        # My residual check computes C[i, k] = sum_j sum_r ...
        # So C[i, k] should be sum_j 1 = 4 for all i, k.
        
        res = np.max(np.abs(C - 4))
        
        if res < 1e-9:
            return {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }
        else:
            # If it fails, return schoolbook
            return fallback
            
    except Exception as e:
        return fallback

# EVOLVE-BLOCK-END
