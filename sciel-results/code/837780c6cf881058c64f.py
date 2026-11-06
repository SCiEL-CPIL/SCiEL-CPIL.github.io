<<<<<<< SEARCH
    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        C6: u-v-x-y-z-w-u where all vertices distinct.
        x in N(v)\{u}, w in N(u)\{v}, y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z.
        """
        Nv = adj[v] & ~(1 << u)
        Nu = adj[u] & ~(1 << v)
        
        # Iterate over the smaller neighbor set first
        if bin(Nv).count('1') <= bin(Nu).count('1'):
            outer_mask = Nv
            inner_mask_template = Nu
            outer_is_v = True
        else:
            outer_mask = Nu
            inner_mask_template = Nv
            outer_is_v = False
        
        while outer_mask:
            outer = (outer_mask & -outer_mask).bit_length() - 1
            outer_mask &= outer_mask - 1
            
            inner_mask = inner_mask_template & ~(1 << outer)
            while inner_mask:
                inner = (inner_mask & -inner_mask).bit_length() - 1
                inner_mask &= inner_mask - 1
                
                if outer_is_v:
                    x, w = outer, inner
                else:
                    x, w = inner, outer
                
                # S_x = N(x) \ {u, v, w}
                S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                # S_w = N(w) \ {u, v, x}
                S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                # Check if there is an edge between S_x and S_w with y != z
                y_mask = S_x
                while y_mask:
                    y = (y_mask & -y_mask).bit_length() - 1
                    y_mask &= y_mask - 1
                    
                    if adj[y] & (S_w & ~(1 << y)):
                        return False
        return True
=======
    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        C6: u-v-x-y-z-w-u where all vertices distinct.
        x in N(v)\{u}, w in N(u)\{v}, y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z.
        """
        Nv = adj[v] & ~(1 << u)
        Nu = adj[u] & ~(1 << v)
        
        # Iterate over the smaller neighbor set first
        if bin(Nv).count('1') <= bin(Nu).count('1'):
            outer_mask = Nv
            inner_mask_template = Nu
            outer_is_v = True
        else:
            outer_mask = Nu
            inner_mask_template = Nv
            outer_is_v = False
        
        while outer_mask:
            outer = (outer_mask & -outer_mask).bit_length() - 1
            outer_mask &= outer_mask - 1
            
            inner_mask = inner_mask_template & ~(1 << outer)
            while inner_mask:
                inner = (inner_mask & -inner_mask).bit_length() - 1
                inner_mask &= inner_mask - 1
                
                if outer_is_v:
                    x, w = outer, inner
                else:
                    x, w = inner, outer
                
                # S_x = N(x) \ {u, v, w}
                S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                # S_w = N(w) \ {u, v, x}
                S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                # Quick check: if S_x and S_w are disjoint, no C6 through this pair
                if not (S_x & S_w):
                    continue
                    
                # Check if there is an edge between S_x and S_w with y != z
                y_mask = S_x
                while y_mask:
                    y = (y_mask & -y_mask).bit_length() - 1
                    y_mask &= y_mask - 1
                    
                    if adj[y] & (S_w & ~(1 << y)):
                        return False
        return True
>>>>>>> REPLACE

