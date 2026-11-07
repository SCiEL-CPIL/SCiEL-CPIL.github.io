import time

def construct():
    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        mask = [0] * n
        for v in range(n):
            m = 0
            for c in lists[v]:
                m |= (1 << c)
            mask[v] = m

        # CBJ Variables
        color = [None] * n
        # conflict_set[v] is a set of variables that caused v to fail or are in the path
        # Standard CBJ:
        # When variable v fails, it backjumps to the earliest variable in its conflict set.
        
        # For simplicity and speed on small graphs, we will use **Non-Chronological Backtracking**
        # which is a form of CBJ.
        
        # However, implementing CBJ correctly is complex.
        # Let's stick to **Improved Backtracking with Value Ordering**.
        # The current code uses MRV.
        # Let's add **Least Constraining Value (LCV)** heuristic.
        # This is a structural change in the *value selection* strategy.
        
        # Current: tries colors in bit order.
        # New: tries colors that constrain neighbors the least.
        
        # This requires calculating the impact of each color.
        
        def backtrack(depth=0):
            if all(color[v] is not None for v in range(n)):
                return True

            # MRV
            v = None
            min_count = 6
            for u in range(n):
                if color[u] is None:
                    cnt = bin(mask[u]).count('1')
                    if cnt < min_count:
                        min_count = cnt
                        v = u
                        if cnt == 0:
                            return False
            
            if v is None:
                return True

            # LCV: Order colors by how many values they remove from neighbors
            colors_in_v = []
            m = mask[v]
            while m:
                c_bit = m & (-m)
                c = c_bit.bit_length() - 1
                colors_in_v.append(c)
                m ^= c_bit
            
            # Calculate constraint for each color
            scored_colors = []
            for c in colors_in_v:
                # Count how many unassigned neighbors lose this color
                # Or simply, count how many neighbors have this color in their list
                # If a neighbor has this color, assigning c to v removes it from neighbor.
                # We want to minimize the number of removals?
                # LCV: choose the value that rules out the fewest choices in neighbors.
                removals = 0
                for u in adj[v]:
                    if color[u] is None:
                        if mask[u] & (1 << c):
                            removals += 1
                scored_colors.append((removals, c))
            
            scored_colors.sort()
            
            for _, c in scored_colors:
                c_bit = 1 << c
                color[v] = c
                
                # Forward Checking
                removed = []
                failed = False
                for u in adj[v]:
                    if color[u] is None:
                        old = mask[u]
                        new = old & ~c_bit
                        if new != old:
                            mask[u] = new
                            removed.append((u, old))
                            if new == 0:
                                failed = True
                                break
                
                if not failed:
                    if backtrack(depth + 1):
                        return True
                
                # Undo
                color[v] = None
                for u, old in removed:
                    mask[u] = old

            return False

        if backtrack():
            return [color[v] for v in range(n)]
        else:
            return [0] * n

    out = []
    start_time = time.time()
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() - start_time > 110:
            # Safety: If we run out of time, we must still return 4 lists.
            # We generate the instance to get n, then fill with 0s?
            # Or just break and pad.
            # But we need the instance to know n.
            # Let's just generate and solve quickly?
            # If we are at 110s, we have 10s left.
            # If we can't solve, we return invalid.
            # Better to not break unless absolutely necessary.
            pass
            
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        out.append([int(c) for c in sol])
        
    while len(out) < 4:
        out.append([0]*100)
        
    return out
