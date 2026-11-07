import sys
sys.setrecursionlimit(10000)

def construct():
    n = 12
    lcm_period = 27720
    
    # Candidate m's: multiples of 2310 (lcm of primes 2,3,5,7,11)
    # and maybe some offsets
    candidates = []
    for k in range(1, 13):
        candidates.append(k * 2310)
    
    # Add some randoms or structured ones
    # Also check m=0
    candidates.append(0)
    
    # Let's also check m such that m+1 is 1 mod p for all primes
    # This is m = 0 mod p. So multiples of 2310.
    
    best_m = 0
    best_L = 0
    
    def compute_L(m):
        # Check L from 12 to 144
        for L in range(12, 145):
            if check_match(m, L):
                return L
        return 144

    def check_match(m, L):
        # Candidates are m+1 to m+L
        # We need to match i in 1..12 to distinct c in candidates such that i|c
        # Use DFS matching
        # Left: 1..12
        # Right: 0..L-1 (index in candidates)
        
        # Precompute divisibility
        # div[i][j] = True if (m+1+j) % i == 0
        # To speed up, we can just check on the fly in DFS
        
        match_to = [-1] * L
        match_from = [-1] * n
        
        def try_match(i, visited):
            for j in range(L):
                if visited[j]:
                    continue
                if (m + 1 + j) % (i + 1) != 0:
                    continue
                visited[j] = True
                if match_to[j] == -1 or try_match(match_to[j], visited):
                    match_to[j] = i
                    match_from[i] = j
                    return True
            return False
            
        found = 0
        for i in range(n):
            visited = [False] * L
            if try_match(i, visited):
                found += 1
            else:
                return False
        return True

    for m in candidates:
        L = compute_L(m)
        if L > best_L:
            best_L = L
            best_m = m
            
    # If best_L is still low, we might need to search more
    # But let's assume one of these is good.
    
    return best_m
