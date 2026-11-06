    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return True
        for i, y in enumerate(S_list):
            # Build DP for S \ {y}
            elements = [x for j, x in enumerate(S_list) if j != i]
            m = len(elements)
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in elements:
                for k in range(m, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
            # Check
            for k in range(2, m + 1):
                if (dp[k] >> (k * y)) & 1:
                    return False
        return True
    