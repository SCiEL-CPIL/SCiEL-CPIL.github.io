def construct():
    N = 12
    
    # a: cycle 0 -> 1 -> ... -> N-1 -> 0
    a_map = [(i + 1) % N for i in range(N)]
    
    # b: Modified Černý / Pawlik-like structure
    # 0 -> 0
    # 1 -> 0
    # i -> i-1 for 2 <= i <= N-2
    # N-1 -> N-1
    b_map = [0] * N
    b_map[0] = 0
    b_map[1] = 0
    for i in range(2, N - 1):
        b_map[i] = i - 1
    b_map[N - 1] = N - 1
    
    return [a_map, b_map]
