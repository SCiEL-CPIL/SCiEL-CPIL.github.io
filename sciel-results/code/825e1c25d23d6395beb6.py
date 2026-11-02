    def _build_m55_rank46():
        # Returns U, V, W for 5x5x5 with rank 46
        # ...
        
    def construct():
        U55, V55, W55 = _build_m55_rank46()
        # Slice for 2x5x5
        U = U55[:10] # 2 rows * 5 cols
        V = V55      # 5x5
        W = []
        for k in range(5):
            for i in range(2):
                W.append(W55[k*5 + i])
        return {'U': U, 'V': V, 'W': W}
    