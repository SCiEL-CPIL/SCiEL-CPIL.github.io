import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 118.0

    def verify(n, colors):
        """Verify no color class contains a C6. Returns True if valid."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[c][i] >> j) & 1:
                        continue
                    found = False
                    for a in range(n):
                        if a == i or a == j:
                            continue
                        if not (adj[c][i] >> a) & 1:
                            continue
                        for b in range(n):
                            if b in (i, j, a):
                                continue
                            if not (adj[c][a] >> b) & 1:
                                continue
                            for d in range(n):
                                if d in (i, j, a, b):
                                    continue
                                if not (adj[c][b] >> d) & 1:
                                    continue
                                for e in range(n):
                                    if e in (i, j, a, b, d):
                                        continue
                                    if not (adj[c][d] >> e) & 1:
                                        continue
                                    if (adj[c][e] >> j) & 1:
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        return False
        return True

    def count_c6_violations(n, colors):
        """Count number of monochromatic C6 in the coloring."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        count = 0
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[c][i] >> j) & 1:
                        continue
                    for a in range(n):
                        if a == i or a == j:
                            continue
                        if not (adj[c][i] >> a) & 1:
                            continue
                        for b in range(n):
                            if b in (i, j, a):
                                continue
                            if not (adj[c][a] >> b) & 1:
                                continue
                            for d in range(n):
                                if d in (i, j, a, b):
                                    continue
                                if not (adj[c][b] >> d) & 1:
                                    continue
                                for e in range(n):
                                    if e in (i, j, a, b, d):
                                        continue
                                    if not (adj[c][d] >> e) & 1:
                                        continue
                                    if (adj[c][e] >> j) & 1:
                                        count += 1
        return count

    def local_search(n, time_budget):
        """Simulated annealing / local search to minimize C6 violations."""
        deadline = time.time() + time_budget
        num_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initialize with random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        current_violations = count_c6_violations(n, colors)
        best_colors = colors[:]
        best_violations = current_violations
        
        if best_violations == 0:
            return best_colors
        
        temp = 10.0
        cooling = 0.9995
        
        while time.time() < deadline:
            # Pick a random edge to flip
            idx = random.randint(0, num_edges - 1)
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            colors[idx] = new_c
            new_violations = count_c6_violations(n, colors)
            
            delta = new_violations - current_violations
            
            if delta <= 0 or random.random() < (temp / max(1, abs(delta))):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
            else:
                colors[idx] = old_c
            
            temp *= cooling
            if temp < 0.01:
                temp = 10.0
        
        return best_colors if best_violations < current_violations else best_colors

    def random_restart_search(n, time_budget):
        """Multiple random restarts with local search."""
        deadline = time.time() + time_budget
        best_result = None
        best_violations = float('inf')
        
        while time.time() < deadline:
            result = local_search(n, min(5.0, deadline - time.time()))
            if result:
                v = count_c6_violations(n, result)
                if v == 0:
                    return result
                if v < best_violations:
                    best_violations = v
                    best_result = result
        
        return best_result

    # Strategy: Try to find valid colorings for increasing n
    # Use random restart local search
    
    # First try n=17
    if time.time() < hard_deadline - 10:
        res = random_restart_search(17, 50)
        if res:
            if verify(17, res):
                return {'n': 17, 'colors': res}
    
    # Try n=16
    if time.time() < hard_deadline - 20:
        res = random_restart_search(16, 25)
        if res:
            if verify(16, res):
                return {'n': 16, 'colors': res}
    
    # Try n=15
    if time.time() < hard_deadline - 10:
        res = random_restart_search(15, 12)
        if res:
            if verify(15, res):
                return {'n': 15, 'colors': res}
    
    # Try n=14
    if time.time() < hard_deadline - 5:
        res = random_restart_search(14, 6)
        if res:
            if verify(14, res):
                return {'n': 14, 'colors': res}
    
    # Try n=13
    if time.time() < hard_deadline - 3:
        res = random_restart_search(13, 3)
        if res:
            if verify(13, res):
                return {'n': 13, 'colors': res}
    
    # Fallback: n=9 with a known valid pattern
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    # Absolute fallback n=5
    n = 5
    colors = [(i * 3 + j * 2) % 4 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
