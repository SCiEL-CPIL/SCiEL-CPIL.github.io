import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def has_k222(colors, n, color):
        # Build adjacency matrix for the specified color
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color:
                    adj[i][j] = 1
                    adj[j][i] = 1
                idx += 1
        
        # Check for K2,2,2
        # For each pair (u1, u2), find common neighbors S.
        # If |S| >= 4, check if S contains a K2,2.
        # A K2,2 exists in S if there are two vertices v1, v2 in S with >= 2 common neighbors in S.
        for u1 in range(n):
            for u2 in range(u1+1, n):
                S = []
                for v in range(n):
                    if v != u1 and v != u2 and adj[u1][v] and adj[u2][v]:
                        S.append(v)
                if len(S) < 4:
                    continue
                
                # Check for K2,2 in S
                # For each pair (v1, v2) in S, count common neighbors in S.
                # Optimization: Precompute common neighbor counts? 
                # For small S, direct iteration is fine.
                k = len(S)
                found = False
                for a in range(k):
                    v1 = S[a]
                    for b in range(a+1, k):
                        v2 = S[b]
                        common = 0
                        for c in range(k):
                            w = S[c]
                            if w != v1 and w != v2 and adj[v1][w] and adj[v2][w]:
                                common += 1
                                if common >= 2:
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    return True
        return False

    def count_k222(colors, n, color):
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color:
                    adj[i][j] = 1
                    adj[j][i] = 1
                idx += 1
        
        count = 0
        for u1 in range(n):
            for u2 in range(u1+1, n):
                S = []
                for v in range(n):
                    if v != u1 and v != u2 and adj[u1][v] and adj[u2][v]:
                        S.append(v)
                if len(S) < 4:
                    continue
                
                k = len(S)
                for a in range(k):
                    v1 = S[a]
                    for b in range(a+1, k):
                        v2 = S[b]
                        common = 0
                        for c in range(k):
                            w = S[c]
                            if w != v1 and w != v2 and adj[v1][w] and adj[v2][w]:
                                common += 1
                        if common >= 2:
                            count += 1
        return count

    def is_valid(colors, n):
        return not has_k222(colors, n, 0) and not has_k222(colors, n, 1)

    def random_colors(n):
        return [random.randint(0, 1) for _ in range(n*(n-1)//2)]

    def local_search(colors, n, max_iter=5000):
        current_violations = count_k222(colors, n, 0) + count_k222(colors, n, 1)
        if current_violations == 0:
            return colors, 0
        
        best_colors = colors[:]
        best_violations = current_violations
        temp = 1.0
        cooling = 0.999
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random edge to flip
            edge_idx = random.randint(0, n*(n-1)//2 - 1)
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            colors[edge_idx] = new_val
            
            new_violations = count_k222(colors, n, 0) + count_k222(colors, n, 1)
            
            delta = new_violations - current_violations
            if delta < 0 or random.random() < pow(2.71828, -delta/temp):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors, 0
            else:
                colors[edge_idx] = old_val
            
            temp *= cooling
        
        return best_colors, best_violations

    # 1. Try to find a valid coloring for n=29 using local search
    best_n = 17
    best_colors = None
    
    # Baseline: Paley(17)
    squares = {x * x % 17 for x in range(1, 17)}
    p17 = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    if is_valid(p17, 17):
        best_n = 17
        best_colors = p17

    # Attempt to find valid coloring for n=29
    if time.time() < deadline - 20:
        # Multiple restarts for n=29
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            colors = random_colors(29)
            colors, viol = local_search(colors, 29, max_iter=2000)
            if viol == 0:
                best_n = 29
                best_colors = colors
                break
        
        # If we found 29, try n=30?
        # The task says "Maximize n". Record has 29. 30 would prove R=31.
        # But finding a valid 30 is extremely hard.
        # Let's stick to 29 as the target if we can find it.
        
    # If we didn't find 29, try 25?
    if best_n < 25 and time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            colors = random_colors(25)
            colors, viol = local_search(colors, 25, max_iter=2000)
            if viol == 0:
                best_n = 25
                best_colors = colors
                break

    if best_colors is None:
        return {'n': 17, 'colors': p17}
        
    return {'n': best_n, 'colors': best_colors}
