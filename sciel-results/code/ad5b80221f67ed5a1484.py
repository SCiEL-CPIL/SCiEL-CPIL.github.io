    for t in range(4):
        b00_row = 2 * t          # V row for B[0, 2t]
        b01_row = 2 * t + 1      # V row for B[0, 2t+1]
        b10_row = 8 + 2 * t      # V row for B[1, 2t]
        b11_row = 8 + 2 * t + 1  # V row for B[1, 2t+1]
        
        c00_row = (2 * t) * 3 + 0      # W row for C[0, 2t]
        c01_row = (2 * t + 1) * 3 + 0  # W row for C[0, 2t+1]
        c10_row = (2 * t) * 3 + 1      # W row for C[1, 2t]
        c11_row = (2 * t + 1) * 3 + 1  # W row for C[1, 2t+1]
