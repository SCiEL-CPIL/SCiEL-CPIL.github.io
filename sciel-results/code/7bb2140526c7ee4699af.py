N = 12

# EVOLVE-BLOCK-START
def construct():
    # Pinwheel construction: 'a' is the cycle, 'b' is shift-left with 0 fixed
    # a: i -> (i+1) % N
    # b: 0 -> 0, i -> i-1 for i >= 1
    a = [(i + 1) % N for i in range(N)]
    b = [0, 0, 0] + [i - 2 for i in range(3, N)]
    return [a, b]
# EVOLVE-BLOCK-END
