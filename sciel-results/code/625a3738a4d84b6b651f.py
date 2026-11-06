def construct():
    import time
    deadline = time.time() + 75
    
    n = 128
    from math import gcd
    
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    conflict_pairs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
    
    degree = [len(conflict_pairs[x]) for x in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: (degree[x], x))
    
    selected = []
    selected_pairs = set()
    
    for x in order:
        if time.time() > deadline:
            break
        can_add = True
        for pair in conflict_pairs[x]:
            if pair in selected_pairs:
                can_add = False
                break
        if can_add:
            for a in selected:
                selected_pairs.add((min(a, x), max(a, x)))
            selected.append(x)
    
    return selected if selected else [1]
