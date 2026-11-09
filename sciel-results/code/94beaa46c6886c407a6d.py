import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 108.0
    MARGIN = 6.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    def verify(n, colors):
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[c][u] & adj[c][v]).bit_count() >= 2:
                        return False
        return True

    def extract(color, n):
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                out.append(color[i][j])
        return out

    def greedy_construct(n, seed=None):
        if seed is not None:
            random.seed(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        total = len(pairs)
        color = [[0] * n for _ in range(n)]
        adj = [[0] * n for _ in range(5)]
        # common[c][u][v] = number of common neighbors of u and v in color c
        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        # Track forbidden pairs: pairs with exactly 1 common neighbor (risky)
        # We'll use a penalty function based on common counts

        for (a, b) in pairs:
            best_c = -1
            best_penalty = float('inf')
            # Try all 5 colors, compute penalty
            for c in range(5):
                # If we color (a,b) with c, what's the penalty?
                # Penalty = sum over all x in N_c(a) of [common_c(b,x) + 1]
                #          + sum over all y in N_c(b) of [common_c(a,y) + 1]
                # But more precisely, we care about how many new "forbidden" pairs we create
                # A pair (u,v) becomes forbidden if common_c[u][v] goes from 1 to 2
                # And a pair becomes "safe" if it goes from 2 to 3 (but that's already a violation)
                
                Na = adj[c][a]
                Nb = adj[c][b]
                penalty = 0.0
                
                # For each neighbor x of a in color c (excluding b):
                #   common_c[b][x] will increase by 1
                #   If common_c[b][x] was 1, it becomes 2 -> creates a violation (bad)
                #   If common_c[b][x] was 0, it becomes 1 -> creates a forbidden pair (less bad)
                x = Na & ~(1 << b)
                while x:
                    lsb = x & (-x)
                    xi = lsb.bit_length() - 1
                    x ^= lsb
                    old = common[c][b][xi]
                    if old == 1:
                        penalty += 100.0  # Would create a C4 violation
                    elif old == 0:
                        penalty += 1.0    # Would create a forbidden pair
                    # if old >= 2, it's already a violation, adding more doesn't change status
                
                # For each neighbor y of b in color c (excluding a):
                x = Nb & ~(1 << a)
                while x:
                    lsb = x & (-x)
                    xi = lsb.bit_length() - 1
                    x ^= lsb
                    old = common[c][a][xi]
                    if old == 1:
                        penalty += 100.0
                    elif old == 0:
                        penalty += 1.0
                
                # Add small random tiebreaker
                penalty += random.random() * 0.1
                
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_c = c
            
            # Assign best_c
            c = best_c
            color[a][b] = c
            color[b][a] = c
            
            # Update adj and common
            Na = adj[c][a]
            Nb = adj[c][b]
            
            # For each x in N_c(a) (excluding b):
            x = Na & ~(1 << b)
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                common[c][b][xi] += 1
                common[c][xi][b] += 1
            
            # For each y in N_c(b) (excluding a):
            x = Nb & ~(1 << a)
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                common[c][a][xi] += 1
                common[c][xi][a] += 1
            
            adj[c][a] |= (1 << b)
            adj[c][b] |= (1 << a)
        
        # Check if valid
        viol = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if common[c][u][v] >= 2:
                        viol += 1
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        if viol == 0:
            cols = extract(color, n)
            if verify(n, cols):
                return cols
        return None

    def local_search(n, init_colors, budget):
        """Local search starting from init_colors (list of n*(n-1)//2 colors)"""
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        total = len(pairs)
        color = [[0] * n for _ in range(n)]
        idx = 0
        for (i, j) in pairs:
            color[i][j] = init_colors[idx]
            idx += 1
        
        adj = [[0] * n for _ in range(5)]
        for (i, j) in pairs:
            c = color[i][j]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        viol = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cc = (adj[c][u] & adj[c][v]).bit_count()
                    common[c][u][v] = cc
                    common[c][v][u] = cc
                    if cc >= 2:
                        viol += 1
        
        if viol == 0:
            cols = extract(color, n)
            if verify(n, cols):
                return cols
        
        T = 1.0
        T_end = 0.005
        
        while time_left() > MARGIN:
            a, b = pairs[random.randrange(total)]
            c1 = color[a][b]
            c2 = random.randrange(5)
            if c2 == c1:
                continue
            
            # Remove from c1
            delta = 0
            Na = adj[c1][a]
            Nb = adj[c1][b]
            x = (Nb & ~(1 << a))
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c1][a][xi]
                new = old - 1
                common[c1][a][xi] = new
                common[c1][xi][a] = new
                if old == 2:
                    delta -= 1
            x = (Na & ~(1 << b))
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c1][b][xi]
                new = old - 1
                common[c1][b][xi] = new
                common[c1][xi][b] = new
                if old == 2:
                    delta -= 1
            adj[c1][a] &= ~(1 << b)
            adj[c1][b] &= ~(1 << a)
            
            # Add to c2
            Na = adj[c2][a]
            Nb = adj[c2][b]
            x = (Nb & ~(1 << a))
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c2][a][xi]
                new = old + 1
                common[c2][a][xi] = new
                common[c2][xi][a] = new
                if old == 1:
                    delta += 1
            x = (Na & ~(1 << b))
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c2][b][xi]
                new = old + 1
                common[c2][b][xi] = new
                common[c2][xi][b] = new
                if old == 1:
                    delta += 1
            adj[c2][a] |= (1 << b)
            adj[c2][b] |= (1 << a)
            
            viol += delta
            
            if delta <= 0 or random.random() < (2.718281828 ** (-delta / T)):
                color[a][b] = c2
            else:
                # Revert
                Na = adj[c2][a]
                Nb = adj[c2][b]
                x = (Nb & ~(1 << a))
                while x:
                    lsb = x & (-x)
                    xi = lsb.bit_length() - 1
                    x ^= lsb
                    old = common[c2][a][xi]
                    new = old - 1
                    common[c2][a][xi] = new
                    common[c2][xi][a] = new
                    if old == 2:
                        viol -= 1
                x = (Na & ~(1 << b))
                while x:
                    lsb = x & (-x)
                    xi = lsb.bit_length() - 1
                    x ^= lsb
                    old = common[c2][b][xi]
                    new = old - 1
                    common[c2][b][xi] = new
                    common[c2][xi][b] = new
                    if old == 2:
                        viol -= 1
                adj[c2][a] &= ~(1 << b)
                adj[c2][b] &= ~(1 << a)
                
                Na = adj[c1][a]
                Nb = adj[c1][b]
                x = (Nb & ~(1 << a))
                while x:
                    lsb = x & (-x)
                    xi = lsb.bit_length() - 1
                    x ^= lsb
                    old = common[c1][a][xi]
                    new = old + 1
                    common[c1][a][xi] = new
                    common[c1][xi][a] = new
                    if old == 1:
                        viol += 1
                x = (Na & ~(1 << b))
                while x:
                    lsb = x & (-x)
                    xi = lsb.bit_length() - 1
                    x ^= lsb
                    old = common[c1][b][xi]
                    new = old + 1
                    common[c1][b][xi] = new
                    common[c1][xi][b] = new
                    if old == 1:
                        viol += 1
                adj[c1][a] |= (1 << b)
                adj[c1][b] |= (1 << a)
            
            T = max(T_end, T * 0.9999)
            
            if viol == 0:
                cols = extract(color, n)
                if verify(n, cols):
                    return cols
        
        return None

    best_n = 0
    best_colors = None

    # Try greedy construction for n=26, 27, 28
    for target in [26, 27, 28]:
        if time_left() <= MARGIN + 1:
            break
        
        # Try multiple greedy attempts with different seeds
        found = None
        for attempt in range(50):
            if time_left() <= MARGIN + 1:
                break
            cols = greedy_construct(target, seed=attempt)
            if cols is not None:
                found = cols
                break
        
        if found is not None:
            best_n = target
            best_colors = found
            continue
        
        # If greedy failed, try local search from random init
        if time_left() <= MARGIN + 5:
            break
        init = [random.randrange(5) for _ in range(target * (target - 1) // 2)]
        result = local_search(target, init, None)
        if result is not None:
            best_n = target
            best_colors = result
            continue

    if best_n == 0 or best_colors is None:
        # Fallback: known valid n=11 cyclic coloring
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    return {'n': best_n, 'colors': best_colors}
