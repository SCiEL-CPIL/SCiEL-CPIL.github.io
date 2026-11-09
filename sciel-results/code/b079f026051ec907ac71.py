import time

def construct():
    # Target n=32
    # Construction: 8 disjoint copies of K4
    # K4 is C5-free.
    # Disjoint union of C5-free graphs is C5-free.
    # Independence number of K4 is 1.
    # Independence number of disjoint union is sum of independence numbers.
    # alpha = 8 * 1 = 8.
    # 8 < 9, so condition satisfied.
    
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
