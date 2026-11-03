def get_3x2x2_block(i0, j0, k0):
    terms = []
    # Strassen for i in {i0, i0+1}
    # Reuse get_strassen_block logic but adapted
    # The existing get_strassen_block(i0, j0, k0) assumes i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # It returns 7 terms.
    strassen_terms = get_strassen_block(i0, j0, k0)
    terms.extend(strassen_terms)
    
    # Schoolbook for i in {i0+2}
    # j in {j0, j0+1}, k in {k0, k0+1}
    # 2 terms
    for j in [j0, j0+1]:
        for k in [k0, k0+1]:
            u = [((i0+2)*m + j, 1)]
            v = [(j*p + k, 1)]
            w = [(k*n + (i0+2), 1)]
            terms.append((u, v, w))
            
    return terms
