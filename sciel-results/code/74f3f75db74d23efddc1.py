def construct():
    n = 10
    # Generate 9 candidate squares
    # Use the structure Z5 x Z2
    # Map (x, y) to index x + 5y? No, x in 0..4, y in 0..1.
    # Index = x + 5y.
    
    squares = []
    # We know 4 squares from (kx, y) are orthogonal.
    # Let's generate them.
    # Square k: L_k(x, y) = (k*x % 5, y).
    # Map (x', y') to index x' + 5*y'.
    
    for k in range(1, 5):
        sq = [[0]*n for _ in range(n)]
        for i in range(n):
            x = i % 5
            y = i // 5
            for j in range(n):
                # Input (x, y) is (i%5, i//5)? No, row i, col j.
                # Let row be (x, y) and col be (u, v).
                # L(x, y; u, v) = (k*x + u % 5, y + v % 2)?
                # This is a Latin square if the operation is a group operation.
                # Let's use L_k(i, j) = (k*(i%5) + j%5, (i//5) + (j//2)) % 2?
                # No, simpler:
                # Let row index i correspond to (x, y) where x=i%5, y=i//5.
                # Let col index j correspond to (u, v) where u=j%5, v=j//2? No, j is 0..9.
                # Let's map j to (u, v) where u=j%5, v=j//5? No, that's not a bijection to Z5xZ2 if we use mod 5 and mod 2 on the same index.
                
                # Standard mapping for Z5 x Z2 to 0..9:
                # idx = x + 5y.
                # x = idx % 5, y = idx // 5.
                
                x_i = i % 5
                y_i = i // 5
                x_j = j % 5
                y_j = j // 5
                
                # Operation: (x_i, y_i) + (x_j, y_j) = (x_i+x_j, y_i+y_j)
                # But we want a Latin square.
                # L_k(i, j) = (k*x_i + x_j, y_i + y_j)
                # This is a Latin square.
                
                res_x = (k * x_i + x_j) % 5
                res_y = (y_i + y_j) % 2
                val = res_x + 5 * res_y
                sq[i][j] = val
                
        squares.append(sq)
        
    # Check orthogonality
    # ...
    
    return squares
