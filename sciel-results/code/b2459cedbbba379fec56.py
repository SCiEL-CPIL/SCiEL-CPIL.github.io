        nu = adj[u] & ~(1 << v)
        nu_list = bits(nu)
        for i in range(len(nu_list)):
            a = nu_list[i]
            for j in range(i+1, len(nu_list)):
                b = nu_list[j]
                c_count = (adj[v] & adj[a] & adj[b]).bit_count()
                if c_count >= 2:
                    return False
                    
        nv = adj[v] & ~(1 << u)
        nv_list = bits(nv)
        for i in range(len(nv_list)):
            a = nv_list[i]
            for j in range(i+1, len(nv_list)):
                b = nv_list[j]
                c_count = (adj[u] & adj[a] & adj[b]).bit_count()
                if c_count >= 2:
                    return False
