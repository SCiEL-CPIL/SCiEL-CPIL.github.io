    def has_c5(n, S):
        S_set = set(S)
        adj_mask = [0] * n
        for s in S_set:
            for i in range(n):
                j = (i + s) % n
                if i != j:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        for v1 in range(n):
            m2 = adj_mask[v1]
            while m2:
                lsb2 = m2 & -m2
                v2 = lsb2.bit_length() - 1
                m2 ^= lsb2
                
                m3 = adj_mask[v2] & ~(1 << v1)
                while m3:
                    lsb3 = m3 & -m3
                    v3 = lsb3.bit_length() - 1
                    m3 ^= lsb3
                    
                    m4 = adj_mask[v3] & ~(1 << v1) & ~(1 << v2)
                    while m4:
                        lsb4 = m4 & -m4
                        v4 = lsb4.bit_length() - 1
                        m4 ^= lsb4
                        
                        m5 = adj_mask[v4] & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                        while m5:
                            lsb5 = m5 & -m5
                            v5 = lsb5.bit_length() - 1
                            m5 ^= lsb5
                            
                            if (adj_mask[v1] & (1 << v5)) and v5 != v2:
                                return True
        return False
