    c_u = n_u
    while c_u:
        lsb_u = c_u & (-c_u)
        x2 = lsb_u.bit_length() - 1
        c_u ^= lsb_u
        
        # x4 must be in n_u, distinct from x2
        c_u2 = c_u # remaining bits in n_u after removing x2? 
        # Wait, c_u is n_u without x2? 
        # Yes.
        
        c_v = n_v
        while c_v:
            lsb_v = c_v & (-c_v)
            x3 = lsb_v.bit_length() - 1
            c_v ^= lsb_v
            
            # x5 must be in n_v, distinct from x3
            # c_v is n_v \ {x3}
            
            # We need to iterate x4 from c_u2 and x5 from c_v
            # This is 4 nested loops effectively, but with bit tricks.
            
            # Let's iterate x4
            c_u_temp = c_u
            while c_u_temp:
                lsb_u2 = c_u_temp & (-c_u_temp)
                x4 = lsb_u2.bit_length() - 1
                c_u_temp ^= lsb_u2
                # ...
