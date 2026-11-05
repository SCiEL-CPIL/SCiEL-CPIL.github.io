import time

DEADLINE = 145  # seconds, safety margin before 150s kill

def _time_left():
    return time.time() - _START_TIME < DEADLINE

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()

    # ---- Build the n=32 seed colouring (Greenwood-Gleason doubled) ----
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

    n_seed = 32
    colors_seed = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if i // 16 == j // 16:
                colors_seed.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors_seed.append(3)

    def _validate(colors, n):
        """Check no monochromatic triangle."""
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        for c in range(4):
            edges = adj[c]
            adj_list = {i: set() for i in range(n)}
            for (i, j) in edges:
                adj_list[i].add(j)
                adj_list[j].add(i)
            for i in range(n):
                for j in adj_list[i]:
                    if j > i:
                        for k in adj_list[i] & adj_list[j]:
                            if k != i and k != j:
                                return False
        return True

    def _return_seed():
        return {'n': n_seed, 'colors': colors_seed}

    if not _validate(colors_seed, n_seed):
        return {'n': 1, 'colors': []}

    best_n = n_seed
    best_colors = colors_seed[:]

    def build_masks(colors, n):
        masks = [[0]*4 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        return masks

    masks = build_masks(colors_seed, n_seed)

    def try_extend(n, colors, masks):
        if not _time_left():
            return None, None
            
        assignment = [-1] * n
        S_masks = [0] * 4
        
        def dfs(pos):
            if not _time_left():
                return False
            if pos == n:
                return True
            
            for c in range(4):
                if (S_masks[c] & masks[pos][c]) == 0:
                    assignment[pos] = c
                    S_masks[c] |= (1 << pos)
                    
                    if dfs(pos + 1):
                        return True
                    
                    S_masks[c] ^= (1 << pos)
                    assignment[pos] = -1
            
            return False

        if dfs(0):
            new_colors = colors + assignment
            new_n = n + 1
            new_masks = [[0]*4 for _ in range(new_n)]
            
            for i in range(n):
                for c in range(4):
                    new_masks[i][c] = masks[i][c]
            
            for i in range(n):
                c = assignment[i]
                new_masks[n][c] |= (1 << i)
                new_masks[i][c] |= (1 << n)
                
            return new_colors, new_masks
        else:
            return None, None

    for _ in range(20):
        if not _time_left():
            break
        
        new_colors, new_masks = try_extend(best_n, best_colors, masks)
        if new_colors:
            best_n += 1
            best_colors = new_colors
            masks = new_masks
        else:
            break

    if best_n > 1 and not _validate(best_colors, best_n):
        return _return_seed()

    return {'n': best_n, 'colors': best_colors}
