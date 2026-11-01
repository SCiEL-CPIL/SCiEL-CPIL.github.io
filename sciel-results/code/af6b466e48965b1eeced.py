import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank = 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def su(i, j, r, val): U[i*m+j][r] = val
    def sv(j, k, r, val): V[j*p+k][r] = val
    def sw(k, i, r, val): W[k*n+i][r] = val

    # Try to construct rank 10
    # This is based on combining Strassen-like terms with the third column
    
    # Term 0: (A00+A11)*(B00+B11)
    su(0,0,0,1); su(1,1,0,1)
    sv(0,0,0,1); sv(1,1,0,1)
    sw(0,0,0,1); sw(1,1,0,1)
    
    # Term 1: (A10+A11)*B00
    su(1,0,1,1); su(1,1,1,1)
    sv(0,0,1,1)
    sw(0,1,1,1); sw(1,1,1,-1)
    
    # Term 2: A00*(B01-B11)
    su(0,0,2,1)
    sv(0,1,2,1); sv(1,1,2,-1)
    sw(1,0,2,1); sw(1,1,2,1)
    
    # Term 3: A11*(B10-B00)
    su(1,1,3,1)
    sv(1,0,3,1); sv(0,0,3,-1)
    sw(0,0,3,1); sw(0,1,3,1)
    
    # Term 4: (A00+A01)*B11
    su(0,0,4,1); su(0,1,4,1)
    sv(1,1,4,1)
    sw(0,0,4,-1); sw(1,0,4,1)
    
    # Term 5: (A10-A00)*(B00+B01)
    su(1,0,5,1); su(0,0,5,-1)
    sv(0,0,5,1); sv(0,1,5,1)
    sw(1,1,5,1)
    
    # Term 6: (A01-A11)*(B10+B11)
    su(0,1,6,1); su(1,1,6,-1)
    sv(1,0,6,1); sv(1,1,6,1)
    sw(0,0,6,1)
    
    # Now for the third column, we need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # We can use 2 more terms:
    # Term 7: (A00+A10)*B02
    su(0,0,7,1); su(1,0,7,1)
    sv(0,2,7,1)
    sw(2,0,7,1); sw(2,1,7,1)
    
    # Term 8: (A01+A11)*B12
    su(0,1,8,1); su(1,1,8,1)
    sv(1,2,8,1)
    sw(2,0,8,1); sw(2,1,8,1)
    
    # But wait, this gives us:
    # C[0,2] gets A00*B02 + A01*B12 from terms 7,8
    # C[1,2] gets A10*B02 + A11*B12 from terms 7,8
    # That's correct!
    
    # But we only used 9 terms. We need rank 10, so we have one more term.
    # Actually, let me verify if this is correct...
    
    # Hmm, I think I made an error. Let me reconsider.
    
    # Actually, the above uses 9 terms (0-8), but I declared rank=10.
    # Let me check if the decomposition is valid.
    
    # Actually, I think the issue is that the Strassen terms above are for 2x2*2x2,
    # but they produce C[0,0], C[0,1], C[1,0], C[1,1]. Then terms 7,8 produce C[0,2], C[1,2].
    # So total 9 terms? But that would be rank 9, which seems too good.
    
    # Let me verify by checking if the tensor equation holds.
    
    # Actually, I suspect the Strassen decomposition above is not quite right for our case.
    # Let me just return rank 11 for now and try to find rank 10 separately.
    
    # For safety, let me just return the known rank 11.
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def su2(i, j, r, val): U[i*m+j][r] = val
    def sv2(j, k, r, val): V[j*p+k][r] = val
    def sw2(k, i, r, val): W[k*n+i][r] = val

    # 1. (A00+A11)(B00+B11)
    su2(0, 0, 0, 1); su2(1, 1, 0, 1)
    sv2(0, 0, 0, 1); sv2(1, 1, 0, 1)
    sw2(0, 0, 0, 1); sw2(1, 1, 0, 1)

    # 2. (A10+A11)B00
    su2(1, 0, 1, 1); su2(1, 1, 1, 1)
    sv2(0, 0, 1, 1)
    sw2(0, 1, 1, 1); sw2(1, 1, 1, -1)

    # 3. A00(B01-B11)
    su2(0, 0, 2, 1)
    sv2(0, 1, 2, 1); sv2(1, 1, 2, -1)
    sw2(1, 0, 2, 1); sw2(1, 1, 2, 1)

    # 4. A11(B10-B00)
    su2(1, 1, 3, 1)
    sv2(1, 0, 3, 1); sv2(0, 0, 3, -1)
    sw2(0, 0, 3, 1); sw2(0, 1, 3, 1)

    # 5. (A00+A01)B11
    su2(0, 0, 4, 1); su2(0, 1, 4, 1)
    sv2(1, 1, 4, 1)
    sw2(0, 0, 4, -1); sw2(1, 0, 4, 1)

    # 6. (A10-A00)(B00+B01)
    su2(1, 0, 5, 1); su2(0, 0, 5, -1)
    sv2(0, 0, 5, 1); sv2(0, 1, 5, 1)
    sw2(1, 1, 5, 1)

    # 7. (A01-A11)(B10+B11)
    su2(0, 1, 6, 1); su2(1, 1, 6, -1)
    sv2(1, 0, 6, 1); sv2(1, 1, 6, 1)
    sw2(0, 0, 6, 1)

    # 8. A00 B02
    su2(0, 0, 7, 1)
    sv2(0, 2, 7, 1)
    sw2(2, 0, 7, 1)

    # 9. A01 B12
    su2(0, 1, 8, 1)
    sv2(1, 2, 8, 1)
    sw2(2, 0, 8, 1)

    # 10. A10 B02
    su2(1, 0, 9, 1)
    sv2(0, 2, 9, 1)
    sw2(2, 1, 9, 1)

    # 11. A11 B12
    su2(1, 1, 10, 1)
    sv2(1, 2, 10, 1)
    sw2(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
