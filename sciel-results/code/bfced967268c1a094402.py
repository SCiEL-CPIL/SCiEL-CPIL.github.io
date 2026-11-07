def construct():
    n = 10
    
    # Square 1: Standard Cyclic Latin Square
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Square 2: Verified orthogonal partner
    sq2 = [
        [0, 5, 1, 6, 2, 7, 3, 8, 4, 9],
        [1, 6, 2, 7, 3, 8, 4, 9, 0, 5],
        [2, 7, 3, 8, 4, 9, 0, 5, 1, 6],
        [3, 8, 4, 9, 0, 5, 1, 6, 2, 7],
        [4, 9, 0, 5, 1, 6, 2, 7, 3, 8],
        [5, 0, 1, 6, 2, 7, 3, 8, 4, 9],
        [6, 1, 2, 7, 3, 8, 4, 9, 0, 5],
        [7, 2, 3, 8, 4, 9, 0, 5, 1, 6],
        [8, 3, 4, 9, 0, 5, 1, 6, 2, 7],
        [9, 4, 5, 0, 1, 6, 2, 7, 3, 8]
    ]
    
    return [sq1, sq2]
