def _find_k5_bitset(adj_bits, n):
    if n < 5:
        return None
    
    def bronk(P, X, size, path):
        if size == 5:
            return path[:]
        if P == 0:
            return None
        if size + bin(P).count('1') < 5:
            return None
            
        p = P
        while p:
            lsb = p & -p
            v = lsb.bit_length() - 1
            p ^= lsb
            
            new_P = P & adj_bits[v] & ~lsb
            res = bronk(new_P, 0, size + 1, path + [v])
            if res:
                return res
        
        return None

    initial_P = 0
    for i in range(n):
        if bin(adj_bits[i]).count('1') >= 4:
            initial_P |= (1 << i)
            
    return bronk(initial_P, 0, 0, [])
