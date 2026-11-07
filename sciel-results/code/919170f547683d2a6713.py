import time

def construct():
    n = 10
    squares = []
    
    # Square 1
    L1 = [[0]*n for _ in range(n)]
    # Square 2
    L2 = [[0]*n for _ in range(n)]
    
    for i in range(n):
        r1 = i // 5
        r2 = i % 5
        
        for j in range(n):
            c1 = j // 5
            c2 = j % 5
            
            # L1: s1 = r1+c1, s2 = r2+c2
            s1 = (r1 + c1) % 5
            s2 = (r2 + c2) % 2
            L1[i][j] = s1 + 5 * s2
            
            # L2: s1' = r1+2c1, s2' = c2
            s1_prime = (r1 + 2 * c1) % 5
            s2_prime = c2
            L2[i][j] = s1_prime + 5 * s2_prime
            
    squares.append(L1)
    squares.append(L2)
    
    return squares
