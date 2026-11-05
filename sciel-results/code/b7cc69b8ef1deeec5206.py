    def construct():
        # Generator polynomial coefficients for QR(17)
        # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
        # Bits: 1, 1, 0, 1, 1, 1, 0, 1, 1
        # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8
        g_int = 0
        for i in [0, 1, 3, 4, 5, 7, 8]:
            g_int |= (1 << i)
            
        # Precompute generator matrix rows (cyclic shifts)
        rows = []
        for i in range(9):
            # Rotate left by i
            # In Python, we can do this with bit manipulation
            # Mask for 17 bits
            mask = (1 << 17) - 1
            row = (g_int << i) | (g_int >> (17 - i))
            row &= mask
            rows.append(row)
            
        words = []
        for k in range(512):
            c = 0
            for j in range(9):
                if (k >> j) & 1:
                    c ^= rows[j]
            
            # Calculate parity
            p = c.bit_count() % 2
            
            # Form 18-bit word
            # Parity bit at position 17
            word = c | (p << 17)
            words.append(word)
            
        return words
    