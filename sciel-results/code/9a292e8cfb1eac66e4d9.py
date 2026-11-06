# Let me compute the codewords to embed
Q = {1, 2, 4, 8, 16, 13, 9, 15}
basis = []
v0 = (1 << 17) - 1
basis.append(v0)
v1 = 0
for i in Q:
    v1 |= (1 << i)
for shift in range(1, 9):
    shifted = (v1 >> shift) | ((v1 & ((1 << shift) - 1)) << (17 - shift))
    basis.append(shifted)

words = []
for mask in range(512):
    word = 0
    for i in range(9):
        if (mask >> i) & 1:
            word ^= basis[i]
    parity = word.bit_count() % 2
    extended_word = (word << 1) | parity
    words.append(extended_word)
