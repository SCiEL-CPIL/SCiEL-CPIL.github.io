    def add_terms(U_src, V_src, W_src):
        rank_src = len(U_src[0])
        for c in range(rank_src):
            for row in range(len(U)):
                U[row].append(0)
            # ...
