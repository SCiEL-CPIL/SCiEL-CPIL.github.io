def construct():
    n = 10
    # Generate L0
    L0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    # Generate L1 (multiplier 2, coprime to n-1? No, 2-1=1, coprime to 10)
    L1 = [[(i + 2*j) % n for j in range(n)] for i in range(n)]
    
    # Verify orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            pairs.add((L0[i][j], L1[i][j]))
            
    if len(pairs) == n*n:
        return [L0, L1]
    else:
        return [L0]
