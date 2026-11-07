def construct():
    n = 10
    A = [[0 for _ in range(n)] for _ in range(n)]
    B = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        x = i // 2
        u = i % 2
        for j in range(n):
            y = j // 2
            v = j % 2
            
            # Square A
            a_val = (x + y) % 5
            w_val = (u + v) % 2
            A[i][j] = 2 * a_val + w_val
            
            # Square B
            a_prime_val = (x + 2 * y) % 5
            w_prime_val = u % 2
            B[i][j] = 2 * a_prime_val + w_prime_val
            
    return [A, B]
