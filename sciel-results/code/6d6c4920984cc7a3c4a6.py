import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    def verify(n, orientation):
        # Reconstruct adjacency bitmask
        # out[i] is an integer where bit j is 1 if i -> j
        out = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    out[i] |= (1 << j)
                    # j <- i, so out[j] does not get bit i
                else:
                    # j -> i
                    out[j] |= (1 << i)
        
        # Check every 4-set
        # For each 4-set S, check if there is a vertex v (v not in S? No, v can be anywhere, but v->u implies v!=u)
        # such that v -> u for all u in S.
        # This is equivalent to: S is a subset of out[v].
        # Since v cannot be in S (as v->v is false), we just check all v.
        # If S subset out[v], then v is a common dominator.
        
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        found = False
                        for v in range(n):
                            if (mask & out[v]) == mask:
                                found = True
                                break
                        if not found:
                            return False
        return True

    # Try n=67
    q = 67
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
        
    orientation = []
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in residues:
                orientation.append(0)
            else:
                orientation.append(1)
                
    # Verify to be safe
    if verify(q, orientation):
        return {'n': q, 'orientation': orientation}
        
    # Fallback to n=83 if 67 fails (unlikely)
    q = 83
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
        
    orientation = []
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in residues:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': q, 'orientation': orientation}
