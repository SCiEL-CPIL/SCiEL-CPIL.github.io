import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE_SECONDS = 115

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # Schoolbook fallback
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    def _verify(U, V, W, tol=1e-9):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return np.allclose(T_rec, T, atol=tol)
    
    # Strategy: Decompose M_{3,4,8} into two M_{3,4,4}
    # M_{3,4,8} = M_{3,4,4} (k=0..3) + M_{3,4,4} (k=4..7)
    # Use Strassen for a 2x2x2 sub-block to get M_{3,4,4} rank 47
    
    sub_n, sub_m, sub_p = 3, 4, 4
    # Total schoolbook terms for 3x4x4: 48
    # We replace the 2x2x2 sub-block (8 terms) with 7 Strassen terms
    # So rank of M_{3,4,4} = 48 - 8 + 7 = 47
    
    r_sub = 47
    U_sub = np.zeros((sub_n*sub_m, r_sub))
    V_sub = np.zeros((sub_m*sub_p, r_sub))
    W_sub = np.zeros((sub_p*sub_n, r_sub))
    
    # Fill in schoolbook terms for all indices except the 2x2x2 sub-block
    # Sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    col = 0
    for i in range(sub_n):
        for j in range(sub_m):
            for k in range(sub_p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Skip sub-block, will be handled by Strassen
                U_sub[i*sub_m+j, col] = 1
                V_sub[j*sub_p+k, col] = 1
                W_sub[k*sub_n+i, col] = 1
                col += 1
    
    # Now fill in Strassen terms for the 2x2x2 sub-block
    # Strassen's 7 products for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # 
    # Result:
    # c00 = M1 - M4 + M5 - M7
    # c01 = M4 + M5
    # c10 = M2 + M3
    # c11 = M1 - M2 + M3 + M7
    
    # The 2x2x2 sub-block uses i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*sub_m+j, so for i in {0,1}, j in {0,1}: indices 0,1,4,5
    # V indices: j*sub_p+k, so for j in {0,1}, k in {0,1}: indices 0,1,4,5
    # W indices: k*sub_n+i, so for k in {0,1}, i in {0,1}: indices 0,1,3,4
    
    strassen_col = col  # Should be 40 (48 - 8)
    assert strassen_col == 40, f"Expected 40 schoolbook terms, got {strassen_col}"
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00 at idx 0, a11 at idx 5 (i=1,j=1 -> 1*4+1=5)
    U_sub[0, strassen_col] = 1
    U_sub[5, strassen_col] = 1
    # V: b00 at idx 0, b11 at idx 5 (j=1,k=1 -> 1*4+1=5)
    V_sub[0, strassen_col] = 1
    V_sub[5, strassen_col] = 1
    # W: c00 at idx 0, c11 at idx 4 (k=1,i=1 -> 1*3+1=4)
    W_sub[0, strassen_col] = 1
    W_sub[4, strassen_col] = 1
    strassen_col += 1
    
    # M2 = (a10+a11)b00
    # U: a10 at idx 4 (i=1,j=0 -> 1*4+0=4), a11 at idx 5
    U_sub[4, strassen_col] = 1
    U_sub[5, strassen_col] = 1
    # V: b00 at idx 0
    V_sub[0, strassen_col] = 1
    # W: c10 at idx 3 (k=1,i=0 -> 1*3+0=3), c11 at idx 4
    W_sub[3, strassen_col] = 1
    W_sub[4, strassen_col] = -1
    strassen_col += 1
    
    # M3 = a00(b10-b11)
    # U: a00 at idx 0
    U_sub[0, strassen_col] = 1
    # V: b10 at idx 4 (j=1,k=0 -> 1*4+0=4), b11 at idx 5
    V_sub[4, strassen_col] = 1
    V_sub[5, strassen_col] = -1
    # W: c10 at idx 3, c11 at idx 4
    W_sub[3, strassen_col] = 1
    W_sub[4, strassen_col] = 1
    strassen_col += 1
    
    # M4 = a11(b01-b00)
    # U: a11 at idx 5
    U_sub[5, strassen_col] = 1
    # V: b01 at idx 1 (j=0,k=1 -> 0*4+1=1), b00 at idx 0
    V_sub[1, strassen_col] = 1
    V_sub[0, strassen_col] = -1
    # W: c00 at idx 0, c01 at idx 1 (k=0,i=1 -> 0*3+1=1)
    W_sub[0, strassen_col] = -1
    W_sub[1, strassen_col] = 1
    strassen_col += 1
    
    # M5 = (a00+a01)b11
    # U: a00 at idx 0, a01 at idx 1 (i=0,j=1 -> 0*4+1=1)
    U_sub[0, strassen_col] = 1
    U_sub[1, strassen_col] = 1
    # V: b11 at idx 5
    V_sub[5, strassen_col] = 1
    # W: c00 at idx 0, c01 at idx 1
    W_sub[0, strassen_col] = -1
    W_sub[1, strassen_col] = 1
    strassen_col += 1
    
    # M6 = (a10-a00)(b00+b01)
    # U: a10 at idx 4, a00 at idx 0
    U_sub[4, strassen_col] = 1
    U_sub[0, strassen_col] = -1
    # V: b00 at idx 0, b01 at idx 1
    V_sub[0, strassen_col] = 1
    V_sub[1, strassen_col] = 1
    # W: c11 at idx 4
    W_sub[4, strassen_col] = 1
    strassen_col += 1
    
    # M7 = (a01-a11)(b10+b11)
    # U: a01 at idx 1, a11 at idx 5
    U_sub[1, strassen_col] = 1
    U_sub[5, strassen_col] = -1
    # V: b10 at idx 4, b11 at idx 5
    V_sub[4, strassen_col] = 1
    V_sub[5, strassen_col] = 1
    # W: c00 at idx 0
    W_sub[0, strassen_col] = -1
    strassen_col += 1
    
    assert strassen_col == r_sub, f"Expected {r_sub} total columns, got {strassen_col}"
    
    # Verify M_{3,4,4} decomposition
    if not _verify(U_sub, V_sub, W_sub):
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Construct M_{3,4,8} from two M_{3,4,4}
    r_total = 2 * r_sub
    U_new = np.zeros((n*m, r_total))
    V_new = np.zeros((m*p, r_total))
    W_new = np.zeros((p*n, r_total))
    
    # First half (k=0..3)
    U_new[:, 0:r_sub] = U_sub
    V_new[0:sub_m*sub_p, 0:r_sub] = V_sub
    W_new[0:sub_p*sub_n, 0:r_sub] = W_sub
    
    # Second half (k=4..7)
    U_new[:, r_sub:2*r_sub] = U_sub
    V_new[sub_m*sub_p:m*p, r_sub:2*r_sub] = V_sub
    W_new[sub_p*sub_n:p*n, r_sub:2*r_sub] = W_sub
    
    # Verify final result
    if _verify(U_new, V_new, W_new):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
