    def maximize(S):
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
        return S
