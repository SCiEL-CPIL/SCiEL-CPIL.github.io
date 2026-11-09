    def run_greedy(order, a_first=True):
        S_A = []
        S_B = []
        U_A = set()
        U_B = set()
        D_A = set()
        D_B = set()
        
        for x in order:
            if a_first:
                if can_add(S_A, U_A, D_B, x):
                    S_A, U_A, D_A = add(S_A, U_A, D_A, x)
                    continue
                if can_add(S_B, U_B, D_A, x):
                    S_B, U_B, D_B = add(S_B, U_B, D_B, x)
                    continue
            else:
                if can_add(S_B, U_B, D_A, x):
                    S_B, U_B, D_B = add(S_B, U_B, D_B, x)
                    continue
                if can_add(S_A, U_A, D_B, x):
                    S_A, U_A, D_A = add(S_A, U_A, D_A, x)
                    continue
        return S_A, S_B, len(S_A) + len(S_B)
