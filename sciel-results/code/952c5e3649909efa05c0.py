    basis = []
    word_0 = 0
    for j in Q:
        word_0 |= (1 << j)
        
    # All basis vectors have the same weight (9), so parity bit is always 1
    parity_bit = 1 << 17
    
    current_word = word_0
    basis.append(current_word | parity_bit)
    
    for i in range(1, 9):
        current_word = (current_word >> 1) | ((current_word & 1) << 16)
        basis.append(current_word | parity_bit)
