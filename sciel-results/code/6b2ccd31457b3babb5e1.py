def construct():
    # Thue-Morse sequence provides a low-discrepancy coloring
    # that effectively breaks up arithmetic progressions.
    # t(n) is the parity of the number of 1s in the binary representation of n.
    colors = [0] * 256
    for i in range(256):
        # i corresponds to integer i+1 in the problem statement
        # We use i (0-indexed) for the Thue-Morse value
        # bin(i).count('1') is efficient enough for 256 items
        colors[i] = bin(i).count('1') % 2
    return colors
