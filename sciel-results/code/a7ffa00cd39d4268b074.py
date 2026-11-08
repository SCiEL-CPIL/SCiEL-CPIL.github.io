    beta_school = [
        [1, 1],   # s=0
        [1, 0],   # s=1
        [1, -1],  # s=2
        [-1, 1],  # s=3
        [0, 1],   # s=4
        [1, 1],   # s=5
        [0, 2]    # s=6: B10+B11, for col2: B_row1_col2 + B_row1_col2? No.
                  # Actually, for s=6: (A01-A11)(B10+B11)
    ]
