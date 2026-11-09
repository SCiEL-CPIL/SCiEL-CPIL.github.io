import time
import random

def _edge_index(i, j, n):
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _verify(n, colors):
    if len(colors) != n * (n - 1) // 2:
        return False
    adj = [[False] * n for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[_edge_index(i, j, n)]
            if c < 0 or c > 3:
                return False
            adj[c][i][j] = adj[c][j][i] = True
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[c][i][j]:
                    continue
                for k in range(j + 1, n):
                    if adj[c][i][k] and adj[c][j][k]:
                        return False
    return True

def _fallback_32():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

def _try_extend(n, colors, deadline):
    current_n = n
    current_colors = colors[:]
    
    for target_n in range(current_n + 1, min(current_n + 8, 63)):
        if time.time() > deadline - 5:
            break
        
        success = False
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            new_colors = current_colors[:]
            for i in range(target_n - 1):
                new_colors.append(random.randint(0, 3))
            
            for iteration in range(80000):
                if time.time() > deadline - 5:
                    break
                
                n_cur = target_n
                adj = [[False] * n_cur for _ in range(4)]
                idx = 0
                for i in range(n_cur):
                    for j in range(i + 1, n_cur):
                        c = new_colors[idx]
                        adj[c][i][j] = adj[c][j][i] = True
                        idx += 1
                
                violation = None
                for c in range(4):
                    for i in range(n_cur):
                        for j in range(i + 1, n_cur):
                            if not adj[c][i][j]:
                                continue
                            for k in range(j + 1, n_cur):
                                if adj[c][i][k] and adj[c][j][k]:
                                    violation = (c, i, j, k)
                                    break
                            if violation:
                                break
                        if violation:
                            break
                    if violation:
                        break
                
                if violation is None:
                    success = True
                    break
                
                c, i, j, k = violation
                edge_choice = random.choice([(i, j), (i, k), (j, k)])
                a, b = edge_choice
                if a > b:
                    a, b = b, a
                eidx = _edge_index(a, b, n_cur)
                new_colors[eidx] = random.randint(0, 3)
            
            if success:
                current_n = target_n
                current_colors = new_colors[:]
                break
    
    if current_n > n:
        return (current_n, current_colors)
    return None

def construct():
    deadline = time.time() + 140
    
    def gf_mul(a, b):
        a0, a1 = a % 7, a // 7
        b0, b1 = b % 7, b // 7
        c0 = (a0 * b0 + 3 * a1 * b1) % 7
        c1 = (a0 * b1 + a1 * b0) % 7
        return c0 + 7 * c1
    
    def gf_sub(a, b):
        return ((a % 7) - (b % 7)) % 7 + 7 * (((a // 7) - (b // 7)) % 7)
    
    def gf_pow(base, exp):
        result = 1
        for _ in range(exp):
            result = gf_mul(result, base)
        return result
    
    primitive = None
    for candidate in range(1, 49):
        if gf_pow(candidate, 6) == 1:
            continue
        if gf_pow(candidate, 14) == 1:
            continue
        if gf_pow(candidate, 21) == 1:
            continue
        primitive = candidate
        break
    
    if primitive is None:
        return _fallback_32()
    
    log_table = {}
    x = 1
    for e in range(42):
        log_table[x] = e
        x = gf_mul(x, primitive)
    
    n = 49
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = gf_sub(i, j)
            colors.append(log_table[diff] % 3)
    
    if not _verify(n, colors):
        return _fallback_32()
    
    best_n = n
    best_colors = colors[:]
    
    if time.time() < deadline - 10:
        ext = _try_extend(n, colors, deadline)
        if ext is not None:
            ext_n, ext_colors = ext
            if ext_n > best_n:
                best_n = ext_n
                best_colors = ext_colors
    
    return {'n': best_n, 'colors': best_colors}
