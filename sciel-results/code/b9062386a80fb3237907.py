def get_decomposition(p):
    if p == 1:
        return get_schoolbook_2x2x1()
    elif p == 2:
        return get_strassen_2x2x2()
    else:
        if p % 2 == 0:
            p1 = p // 2
            p2 = p // 2
        else:
            # Find largest power of 2 < p
            p1 = 1
            while p1 * 2 < p:
                p1 *= 2
            p2 = p - p1
            
        U1, V1, W1 = get_decomposition(p1)
        U2, V2, W2 = get_decomposition(p2)
        
        return combine_blocks(U1, V1, W1, U2, V2, W2, p1, p2)
