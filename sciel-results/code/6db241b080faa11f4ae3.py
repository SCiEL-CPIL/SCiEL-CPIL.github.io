import time
from fractions import Fraction

def construct():
        start = time.time()
        deadline = start + 55

        def solve_dp(target_lcm):
            # Find divisors of target_lcm that are >= 5
            candidates = []
            for d in range(5, target_lcm + 1):
                if target_lcm % d == 0:
                    candidates.append(d)
            
            if not candidates:
                return None
                
            INF = float('inf')
            dp = [INF] * (target_lcm + 1)
            parent = [-1] * (target_lcm + 1)
            dp[0] = 0
            
            for d in candidates:
                w = target_lcm // d
                for s in range(target_lcm, w - 1, -1):
                    if dp[s - w] + 1 < dp[s]:
                        dp[s] = dp[s - w] + 1
                        parent[s] = d
            
            if dp[target_lcm] == INF:
                return None
                
            result = []
            s = target_lcm
            while s > 0:
                d = parent[s]
                if d == -1:
                    return None
                result.append(d)
                s -= target_lcm // d
                
            return result

        # Try a list of LCMs
        # 2520, 27720, 30030, 55440, 277200
        lcms = [2520, 27720, 30030, 55440]
        
        for lcm in lcms:
            if time.time() > deadline:
                break
            result = solve_dp(lcm)
            if result is not None:
                if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                    if abs(sum(Fraction(1, d) for d in result) - 1) < 1e-9:
                        return result

        # Fallback to DFS (from current program)
        # ... (include the DFS code)
        
        raise RuntimeError("no baseline representation")
