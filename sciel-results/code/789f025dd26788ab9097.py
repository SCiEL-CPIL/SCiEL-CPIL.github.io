N = 16

def construct():
    colors = [0] * (N * (N - 1) // 2)
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = i
            if i == 13 and j == 14:
                idx_13_14 = idx
            if i == 14 and j == 15:
                idx_14_15 = idx
            idx += 1
            
    colors[idx_13_14] = 16
    colors[idx_14_15] = 15
    
    return colors
