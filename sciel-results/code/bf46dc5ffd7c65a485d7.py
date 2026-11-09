<<<<<<< SEARCH
    wires = [0] * N
    for w in range(N):
        # Build the integer where bit k is 1 iff (k >> w) & 1 is 1
        # This is a pattern: 0...0 1...1 0...0 1...1 ... with period 2^(w+1)
        block = (1 << w) - 1  # w ones
        period = 1 << (w + 1)
        # Repeat the pattern 2^(N-w-1) times
        count = 1 << (N - w - 1)
        val = 0
        for _ in range(count):
            val = (val << period) | block
        wires[w] = val
=======
    wires = [0] * N
    for w in range(N):
        # Bit k of wire w's mask is 1 iff (k >> w) & 1 == 1
        # Period = 2^(w+1): lower 2^w bits are 0, upper 2^w bits are 1
        half = 1 << w
        ones_block = ((1 << half) - 1) << half
        period = 1 << (w + 1)
        count = 1 << (N - w - 1)
        val = 0
        for _ in range(count):
            val = (val << period) | ones_block
        wires[w] = val
>>>>>>> REPLACE
