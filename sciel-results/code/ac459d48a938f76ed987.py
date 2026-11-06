def construct():
    # Generator matrix for the [12, 6, 6] binary self-dual code
    # The supports of the weight-6 codewords form the Witt design S(5,6,12)
    G = [
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    ]
    
    blocks = []
    # Iterate over all 2^6 = 64 possible combinations of basis vectors
    for mask in range(64):
        codeword = [0] * 12
        for i in range(6):
            if (mask >> i) & 1:
                # Add row i to codeword (XOR)
                for j in range(12):
                    codeword[j] ^= G[i][j]
        
        # Check if weight is 6
        if sum(codeword) == 6:
            # Get the support (indices of 1s)
            support = tuple(j for j in range(12) if codeword[j] == 1)
            blocks.append(list(support))
            
    return blocks
