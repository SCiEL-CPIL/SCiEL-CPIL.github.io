        def dfs(rem, min_d):
            if rem < 1e-9:
                return True
            if len(path) >= 256:
                return False
            
            k_left = 256 - len(path)
            
            if rem > k_left / min_d + 1e-9:
                return False
            if rem < k_left / M - 1e-9:
                return False

            if k_left == 1:
                d = round(1.0 / rem)
                if min_d < d <= M:
                    if abs(rem - 1.0/d) < 1e-9:
                        # Verify
                        if Fraction(1, d) == target - sum(Fraction(1, x) for x in path):
                            path.append(d)
                            return True
                return False

            d_min = max(min_d + 1, int(1.0 / rem) + 1)
            if d_min > M:
                return False
            
            val = rem - (k_left - 1) / M
            if val > 0:
                d_min = max(d_min, int(1.0 / val) + 1)
            
            for d in range(d_min, M + 1):
                new_rem = rem - 1.0 / d
                if new_rem < -1e-9:
                    continue
                path.append(d)
                if dfs(new_rem, d):
                    return True
                path.pop()
            return False
