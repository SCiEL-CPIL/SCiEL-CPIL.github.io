"""Matrix multiplication 5x5x5 using 2+2+1 block split with Strassen for 2x2x2 blocks."""
DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    terms = []
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                a_s = block_sizes[bi]
                b_s = block_sizes[bj]
                c_s = block_sizes[bk]
                
                if a_s == 2 and b_s == 2 and c_s == 2:
                    # Strassen 2x2x2: 7 terms
                    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
                    terms.append((
                        [(offsets[bi]+0, 1), (offsets[bi]+3, 1)],
                        [(offsets[bj]+0, 1), (offsets[bj]+3, 1)],
                        [(offsets[bk]+0, 1), (offsets[bk]+3, 1)]
                    ))
                    # M2 = (A10+A11)*B00 -> C10 += M2
                    terms.append((
                        [(offsets[bi]+2, 1), (offsets[bi]+3, 1)],
                        [(offsets[bj]+0, 1)],
                        [(offsets[bk]+2, 1)]
                    ))
                    # M3 = A00*(B10-B11) -> C01 += M3
                    terms.append((
                        [(offsets[bi]+0, 1)],
                        [(offsets[bj]+2, 1), (offsets[bj]+3, -1)],
                        [(offsets[bk]+1, 1)]
                    ))
                    # M4 = A11*(B01-B00) -> C00 += M4, C10 += M4
                    terms.append((
                        [(offsets[bi]+3, 1)],
                        [(offsets[bj]+1, 1), (offsets[bj]+0, -1)],
                        [(offsets[bk]+0, 1), (offsets[bk]+2, 1)]
                    ))
                    # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
                    terms.append((
                        [(offsets[bi]+0, 1), (offsets[bi]+1, 1)],
                        [(offsets[bj]+3, 1)],
                        [(offsets[bk]+0, -1), (offsets[bk]+1, 1)]
                    ))
                    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
                    terms.append((
                        [(offsets[bi]+2, 1), (offsets[bi]+0, -1)],
                        [(offsets[bj]+0, 1), (offsets[bj]+1, 1)],
                        [(offsets[bk]+3, 1)]
                    ))
                    # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 -= M7
                    terms.append((
                        [(offsets[bi]+1, 1), (offsets[bi]+3, -1)],
                        [(offsets[bj]+2, 1), (offsets[bj]+3, 1)],
                        [(offsets[bk]+0, 1), (offsets[bk]+3, -1)]
                    ))
                else:
                    for i in range(a_s):
                        for j in range(b_s):
                            for k in range(c_s):
                                terms.append((
                                    [(offsets[bi]+i, 1)],
                                    [(offsets[bj]+j, 1)],
                                    [(offsets[bk]+k, 1)]
                                ))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
