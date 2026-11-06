N = 12

# EVOLVE-BLOCK-START
def construct():
    # We will test a few deterministic variants of the Pinwheel construction.
    # Variant 1: Standard Pinwheel (current)
    # Variant 2: Shift b by 1
    # Variant 3: Shift b by 2
    # We compute the reset length for each and return the best.
    
    def get_reset_length(a, b):
        # BFS on subset automaton
        from collections import deque
        start = tuple(range(N))
        if len(start) == 1:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            s, d = q.popleft()
            for trans in [a, b]:
                next_s = tuple(sorted(set(trans[i] for i in s)))
                if len(next_s) == 1:
                    return d + 1
                if next_s not in visited:
                    visited.add(next_s)
                    q.append((next_s, d + 1))
        return -1 # Not synchronized

    best_a = None
    best_b = None
    best_len = 0
    
    # Generate candidates
    candidates = []
    
    # 1. Standard Pinwheel
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [0] + [i - 1 for i in range(1, N)]
    candidates.append((a1, b1))
    
    # 2. Reverse Pinwheel
    a2 = [(i - 1) % N for i in range(N)]
    b2 = [0] + [i + 1 for i in range(1, N)]
    candidates.append((a2, b2))
    
    # 3. Shifted b
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [1] + [i for i in range(N-1)] # b(0)=1, b(i)=i? No.
    # Let's try b(i) = (i+1)%N for i<N-1, b(N-1)=0
    b3 = [(i + 1) % N for i in range(N-1)] + [0]
    candidates.append((a3, b3))
    
    # 4. Another variant
    a4 = [(i + 1) % N for i in range(N)]
    b4 = [0] + [i - 1 for i in range(1, N-1)] + [N-2]
    candidates.append((a4, b4))
    
    for a, b in candidates:
        l = get_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a
            best_b = b
            
    return [best_a, best_b]
# EVOLVE-BLOCK-END
