N = 40

def construct():
    edges = []
    # Path 1: 0..19
    for i in range(19):
        edges.append((i, i + 1))
    # Path 2: 20..39
    for i in range(20, 39):
        edges.append((i, i + 1))
    # Rungs
    for i in range(20):
        edges.append((i, i + 20))
    # Extra edge for odd cycles
    edges.append((0, 2))
    
    cycles = []
    for L in range(3, N + 1):
        if L == 3:
            cyc = [0, 1, 2]
        elif L % 2 == 0:
            k = L // 2
            part1 = list(range(0, k))
            part2 = list(range(20 + k - 1, 19, -1))
            cyc = part1 + part2
        else:
            k = (L - 1) // 2
            part1 = [0, 2] + list(range(3, k + 1))
            part2 = list(range(20 + k, 19, -1))
            cyc = part1 + part2
        cycles.append(cyc)
        
    return {'edges': edges, 'cycles': cycles}
