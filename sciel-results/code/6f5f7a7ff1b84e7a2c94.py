<<<<<<< SEARCH
    # Block (0,2): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Dimensions: 2x2 * 2x3 -> 2x3
    # Use Schoolbook
    schoolbook_block(0, 2, 0, 2, 2, 5)
=======
    # Block (0,2): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Dimensions: 2x2 * 2x3 -> 2x3
    # Use optimized: Strassen for top-left 2x2*2x2 part + schoolbook for rest
    # C[0:2, 2:5] = A[0:2, 0:2] * B[0:2, 2:5]
    # Split: top-left 2x2*2x2 (Strassen) + remaining
    # Rows 0,1 of C, cols 2,3 of C
    # Use Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_2x2_block(0, 1, 0, 1, 2, 3)
    # Schoolbook for col 4 of C
    for i in [0, 1]:
        for k in [4]:
            for j in [0, 1]:
                add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})
>>>>>>> REPLACE

