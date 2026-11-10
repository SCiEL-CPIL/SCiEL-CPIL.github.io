            if x >= 0:
                for k in range(n - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
                        dp[k+1] &= MASK
                        if k + 1 >= 2 and (dp[k+1] & (1 << OFFSET)):
                            return True
