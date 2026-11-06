def design(problem):
    size = problem["pixels"] # 32
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Centers at +/- 8 from center (15.5)
    # 15.5 - 8 = 7.5 -> pixel 7 or 8?
    # Let's use coordinates i, j from 0 to 31.
    # Center is 15.5.
    # Target centers: 15.5 +/- 8 => 7.5 and 23.5.
    # Let's use 7.5 and 23.5 as the geometric centers of the blobs.
    
    centers = [
        (7.5, 7.5),
        (7.5, 23.5),
        (23.5, 7.5),
        (23.5, 23.5)
    ]
    
    radius_sq = 4.5**2 # 20.25
    
    for i in range(size):
        for j in range(size):
            # Pixel center is at (i+0.5, j+0.5)
            px = i + 0.5
            py = j + 0.5
            for cx, cy in centers:
                dist_sq = (px - cx)**2 + (py - cy)**2
                if dist_sq < radius_sq:
                    cell[i][j] = 1
                    break
    return cell
