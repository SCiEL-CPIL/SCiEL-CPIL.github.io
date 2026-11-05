def construct():
    # ... existing mul/log setup ...
    # Generate colors for K16
    k16_colors = {}
    for i in range(16):
        for j in range(i+1, 16):
            k16_colors[(i,j)] = log[i ^ j] % 3
            
    n = 48
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            block_i = i // 16
            block_j = j // 16
            if block_i == block_j:
                # Local indices
                li = i % 16
                lj = j % 16
                colors.append(k16_colors[(li, lj)])
            else:
                colors.append(3)
                
    return {'n': n, 'colors': colors}
