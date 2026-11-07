"""Rewritten program: minimize max(distinct edge sums, distinct edge products) for 14 values, 35 edges."""
# EVOLVE-BLOCK-START

def construct():
    # Values 1..14 at indices 0..13
    values = list(range(1, 15))
    
    # 35 edges selected to maximize product collisions and minimize distinct sums
    # Product groups (same product on multiple edges):
    # 24: (2,12), (3,8), (4,6) -> indices (1,11),(2,7),(3,5)
    # 12: (2,6), (3,4) -> (1,5),(2,3)
    # 18: (2,9), (3,6) -> (1,8),(2,5)
    # 20: (2,10), (4,5) -> (1,9),(3,4)
    # 30: (3,10), (5,6) -> (2,9),(4,5)
    # 36: (3,12), (4,9) -> (2,11),(3,8)
    # 40: (4,10), (5,8) -> (3,9),(4,7)
    # 42: (6,7), (3,14) -> (5,6),(2,13)
    # 48: (4,12), (6,8) -> (3,11),(5,7)
    # 60: (5,12), (6,10) -> (4,11),(5,9)
    # 14 single-product edges:
    # 54: (6,9) -> (5,8)
    # 56: (7,8) -> (6,7)
    # 63: (7,9) -> (6,8)
    # 66: (6,11) -> (5,10)
    # 70: (7,10) -> (6,9)
    # 72: (8,9) -> (7,8)
    # 77: (7,11) -> (6,10)
    # 80: (8,10) -> (7,9)
    # 84: (7,12) -> (6,11)
    # 88: (8,11) -> (7,10)
    # 90: (9,10) -> (8,9)
    # 91: (7,13) -> (6,12)
    # 96: (8,12) -> (7,11)
    # 99: (9,11) -> (8,10)
    
    edges = [
        [1, 11], [2, 7], [3, 5],       # 24
        [1, 5],  [2, 3],               # 12
        [1, 8],  [2, 5],               # 18
        [1, 9],  [3, 4],               # 20
        [2, 9],  [4, 5],               # 30
        [2, 11], [3, 8],               # 36
        [3, 9],  [4, 7],               # 40
        [5, 6],  [2, 13],              # 42
        [3, 11], [5, 7],               # 48
        [4, 11], [5, 9],               # 60
        [5, 8],                         # 54
        [6, 7],                         # 56
        [6, 8],                         # 63
        [5, 10],                        # 66
        [6, 9],                         # 70
        [7, 8],                         # 72
        [6, 10],                        # 77
        [7, 9],                         # 80
        [6, 11],                        # 84
        [7, 10],                        # 88
        [8, 9],                         # 90
        [6, 12],                        # 91
        [7, 11],                        # 96
        [8, 10],                        # 99
    ]
    
    # Verify: exactly 35 edges, no duplicates, valid indices
    assert len(edges) == 35
    assert all(0 <= e[0] < 14 and 0 <= e[1] < 14 and e[0] != e[1] for e in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == 35
    
    return {'values': values, 'edges': edges}

# EVOLVE-BLOCK-END
