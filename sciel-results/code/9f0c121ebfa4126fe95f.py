import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Try from n=26 down to n=11
    for target_n in range(26, 10, -1):
        if time.time() > deadline - 8:
            break
        result = _solve(target_n, deadline)
        if result is not None:
            return result
    
    # Fallback: cyclic construction for n=11
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}


def _solve(n, deadline):
    """Find valid 5-coloring of K_n with no mono-C4."""
    num_edges = n * (n - 1) // 2
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Multiple restarts
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        random.seed(attempt * 7919 + n)
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build adjacency
        adj = [[set() for _ in range(5)] for _ in range(n)]
        for k in range(num_edges):
            i, j = edge_list[k]
            c = colors[k]
            adj[i][c].add(j)
            adj[j][c].add(i)
        
        # Count C4 violations
        def count_violations():
            cnt = 0
            for c in range(5):
                for i in range(n):
                    si = adj[i][c]
                    for j in range(i + 1, n):
                        if len(si & adj[j][c]) >= 2:
                            cnt += 1
            return cnt
        
        curr = count_violations()
        if curr == 0:
            return {'n': n, 'colors': colors}
        
        # Simulated annealing
        temp = 5.0
        best = curr
        best_colors = colors[:]
        
        while time.time() < deadline - 6:
            # Pick random edge
            k = random.randint(0, num_edges - 1)
            i, j = edge_list[k]
            old_c = colors[k]
            
            # Try changing to a random other color
            new_c = random.choice([c for c in range(5) if c != old_c])
            
            # Apply
            adj[i][old_c].discard(j)
            adj[j][old_c].discard(i)
            adj[i][new_c].add(j)
            adj[j][new_c].add(i)
            colors[k] = new_c
            
            # Recount (fast for small n)
            new_cnt = count_violations()
            
            if new_cnt < best:
                best = new_cnt
                best_colors = colors[:]
                if best == 0:
                    return {'n': n, 'colors': colors}
            elif new_cnt <= curr or random.random() < pow(2.718, (curr - new_cnt) / temp):
                curr = new_cnt
            else:
                # Revert
                colors[k] = old_c
                adj[i][new_c].discard(j)
                adj[j][new_c].discard(i)
                adj[i][old_c].add(j)
                adj[j][old_c].add(i)
            
            temp *= 0.99999
            if temp < 0.5:
                temp = 0.5
        
        if best == 0:
            return {'n': n, 'colors': best_colors}
    
    return None
