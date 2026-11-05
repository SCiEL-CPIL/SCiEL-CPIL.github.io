        nu = adj[u]
        nv = adj[v]
        
        temp_u = nu
        while temp_u:
            lsb = temp_u & -temp_u
            x = lsb.bit_length() - 1
            temp_u ^= lsb
            
            if x == v: continue # x cannot be v
            
            temp_v = nv
            while temp_v:
                lsb_v = temp_v & -temp_v
                y = lsb_v.bit_length() - 1
                temp_v ^= lsb_v
                
                if y == u: continue
                if x == y: continue # Triangle
                
                # Check if (x,y) is an edge
                if (adj[x] >> y) & 1:
                    return False
        
        return True
