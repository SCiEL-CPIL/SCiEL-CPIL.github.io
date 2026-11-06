import time

def construct():
        start_time = time.time()
        deadline = start_time + 55.0
        
        n = 5040
        target = 4979
        
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        
        # Filter out divisors larger than target
        divs = [d for d in divs if d <= target]
        
        # Sort descending (optional for DP correctness, but good for early pruning in some variants)
        divs.sort(reverse=True)
        
        INF = float('inf')
        dp = [INF] * (target + 1)
        dp[0] = 0
        
        parent = [None] * (target + 1)
        
        for d in divs:
            if time.time() > deadline:
                break
                
            # Update from target down to d
            for s in range(target, d - 1, -1):
                if dp[s - d] + 1 < dp[s]:
                    dp[s] = dp[s - d] + 1
                    parent[s] = (s - d, d)
        
        if dp[target] == INF:
            return []
            
        current = target
        result = []
        while current > 0:
            prev, d = parent[current]
            result.append(d)
            current = prev
            
        return result
