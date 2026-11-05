strassen_block(0, 0, 0, idx)      # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
strassen_block(0, 0, 2, idx)      # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
schoolbook_2x2x1_block(0, 0, 4, idx)  # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]

strassen_block(2, 2, 0, idx)      # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
strassen_block(2, 2, 2, idx)      # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
schoolbook_2x2x1_block(2, 2, 4, idx)  # A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
