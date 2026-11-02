    def construct():
        n, m, p = 4, 4, 5
        rank = 76
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        
        def add_strassen(ar, ac, br, bc, cr, cc):
            nonlocal col
            # ... 7 steps ...
            
        def add_naive(ar, ac, br, bc, cr, cc):
            nonlocal col
            # ... 12 steps ...
            
        # ... calls ...
        
        return {"U": U, "V": V, "W": W}
    