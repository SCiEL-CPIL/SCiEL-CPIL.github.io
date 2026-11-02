import time
import sys

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for 3x3x2
    # Strassen terms for 2x2x2 (indices 0-6)
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: 00=1, 11=1
    # V: 00=1, 11=1
    # W: 00=1, 11=1
    # Term 1: M2 = (A10+A11)B00
    # U: 10=1, 11=1
    # V: 00=1
    # W: 10=1
    # Term 2: M3 = A00(B01-B11)
    # U: 00=1
    # V: 01=1, 11=-1
    # W: 01=1
    # Term 3: M4 = A11(B10-B00)
    # U: 11=1
    # V: 10=1, 00=-1
    # W: 10=1
    # Term 4: M5 = (A00+A01)B11
    # U: 00=1, 01=1
    # V: 11=1
    # W: 01=1
    # Term 5: M6 = (A10-A00)(B00+B01)
    # U: 10=1, 00=-1
    # V: 00=1, 01=1
    # W: 11=1
    # Term 6: M7 = (A01-A11)(B10+B11)
    # U: 01=1, 11=-1
    # V: 10=1, 11=1
    # W: 00=1, 11=1
    
    # Wait, I need to check the W signs for Strassen.
    # c00 = m1 - m4 + m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m6 - m7
    
    # So:
    # T0 (M1): W[00]=1, W[11]=1
    # T1 (M2): W[10]=1
    # T2 (M3): W[01]=1
    # T3 (M4): W[10]=1
    # T4 (M5): W[01]=1
    # T5 (M6): W[11]=1
    # T6 (M7): W[00]=-1, W[11]=-1
    
    # Let's verify T3 (M4).
    # M4 = A11(B10-B00).
    # Contribution to c10: +1 * M4.
    # My T3 W[10]=1. Correct.
    # Contribution to c00: -1 * M4.
    # My T3 W[00] should be -1?
    # In Strassen, c00 = m1 - m4 + m5 - m7.
    # So M4 has coefficient -1 in c00.
    # So T3 should have W[00]=-1?
    # But M4 is defined as A11(B10-B00).
    # If I use T3 with W[10]=1 and W[00]=-1, then:
    # c10 += 1 * M4.
    # c00 += -1 * M4.
    # This matches.
    
    # Let's re-verify all W signs for Strassen.
    # T0 (M1): c00=1, c11=1.
    # T1 (M2): c10=1.
    # T2 (M3): c01=1.
    # T3 (M4): c10=1, c00=-1.
    # T4 (M5): c01=1.
    # T5 (M6): c11=1.
    # T6 (M7): c00=-1, c11=-1.
    
    # Check c00: 1(T0) - 1(T3) + 0(T4) - 1(T6) = T0 - T3 - T6.
    # Formula: m1 - m4 - m7. Correct.
    # Check c01: 0(T0) + 0(T1) + 1(T2) + 0(T3) + 1(T4) + 0(T5) + 0(T6) = T2 + T4.
    # Formula: m3 + m5. Correct.
    # Check c10: 0(T0) + 1(T1) + 0(T2) + 1(T3) + 0(T4) + 0(T5) + 0(T6) = T1 + T3.
    # Formula: m2 + m4. Correct.
    # Check c11: 1(T0) + 0(T1) + 0(T2) + 0(T3) + 0(T4) + 1(T5) - 1(T6) = T0 + T5 - T6.
    # Formula: m1 + m6 - m7. Correct.

    # New terms:
    # T7: A02, A12 with B20, B21 -> C00, C01, C10, C11
    # U: 02=1, 12=1
    # V: 20=1, 21=1
    # W: 00=1, 01=1, 10=1, 11=1
    
    # T8: A20 with B00, B01 -> C20, C21
    # U: 20=1
    # V: 00=1, 01=1
    # W: 20=1, 21=1
    
    # T9: A21 with B10, B11 -> C20, C21
    # U: 21=1
    # V: 10=1, 11=1
    # W: 20=1, 21=1
    
    # T10: A22 with B20, B21 -> C20, C21
    # U: 22=1
    # V: 20=1, 21=1
    # W: 20=1, 21=1

    terms = []
    
    # T0
    terms.append({
        'U': [(0,0,1), (1,1,1)],
        'V': [(0,0,1), (1,1,1)],
        'W': [(0,0,1), (1,1,1)]
    })
    # T1
    terms.append({
        'U': [(1,0,1), (1,1,1)],
        'V': [(0,0,1)],
        'W': [(1,0,1)]
    })
    # T2
    terms.append({
        'U': [(0,0,1)],
        'V': [(0,1,1), (1,1,-1)],
        'W': [(0,1,1)]
    })
    # T3
    terms.append({
        'U': [(1,1,1)],
        'V': [(1,0,1), (0,0,-1)],
        'W': [(1,0,1), (0,0,-1)]
    })
    # T4
    terms.append({
        'U': [(0,0,1), (0,1,1)],
        'V': [(1,1,1)],
        'W': [(0,1,1)]
    })
    # T5
    terms.append({
        'U': [(1,0,1), (0,0,-1)],
        'V': [(0,0,1), (0,1,1)],
        'W': [(1,1,1)]
    })
    # T6
    terms.append({
        'U': [(0,1,1), (1,1,-1)],
        'V': [(1,0,1), (1,1,1)],
        'W': [(0,0,-1), (1,1,-1)]
    })
    # T7
    terms.append({
        'U': [(0,2,1), (1,2,1)],
        'V': [(2,0,1), (2,1,1)],
        'W': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    })
    # T8
    terms.append({
        'U': [(2,0,1)],
        'V': [(0,0,1), (0,1,1)],
        'W': [(2,0,1), (2,1,1)]
    })
    # T9
    terms.append({
        'U': [(2,1,1)],
        'V': [(1,0,1), (1,1,1)],
        'W': [(2,0,1), (2,1,1)]
    })
    # T10
    terms.append({
        'U': [(2,2,1)],
        'V': [(2,0,1), (2,1,1)],
        'W': [(2,0,1), (2,1,1)]
    })

    # Fill U, V, W
    for b in range(3):
        for t_idx in range(11):
            col = b * 11 + t_idx
            term = terms[t_idx]
            
            # U
            for i, j, val in term['U']:
                U[i * m + j][col] = val
            
            # V
            for j, k, val in term['V']:
                k_global = 2 * b + k
                V[j * p + k_global][col] = val
            
            # W
            for k, i, val in term['W']:
                k_global = 2 * b + k
                W[k_global * n + i][col] = val

    return {"U": U, "V": V, "W": W}
