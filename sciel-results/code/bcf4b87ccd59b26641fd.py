import time

def construct():
    n = 17
    colors = []
    # Pre-defined mapping of difference d (1..16) to color
    # d is (j - i) % n
    # We need to handle the undirected nature, so we can just use the difference.
    # The sets are:
    # 0: {1, 2, 15, 16}
    # 1: {4, 8, 9, 13}
    # 2: {3, 5, 12, 14}
    # 3: {6, 7, 10, 11}
    
    color_map = {
        1: 0, 2: 0, 15: 0, 16: 0,
        4: 1, 8: 1, 9: 1, 13: 1,
        3: 2, 5: 2, 12: 2, 14: 2,
        6: 3, 7: 3, 10: 3, 11: 3
    }
    
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(color_map[d])
            
    return {'n': n, 'colors': colors}
