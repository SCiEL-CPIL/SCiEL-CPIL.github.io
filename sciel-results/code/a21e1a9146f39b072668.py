"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Known solution: 1/8 + 1/31 + 1/248 = 5/31
    # Largest denominator: 248
    best_solution = [8, 31, 248]
    best_M = 248
    
    # Search for 3-term solutions with largest denominator M < 248
    # 1/a + 1/b + 1/M = 5/31
    # 1/a + 1/b = (5M - 31) / (31M)
    # Let K = 5M - 31. Then 1/a + 1/b = K / (31M)
    # (a+b) / ab = K / (31M) => 31M(a+b) = Kab => 31Ma + 31Mb = Kab
    # 31Ma = b(Ka - 31M) => b = 31Ma / (Ka - 31M)
    
    for M in range(1, 248):
        K = 5 * M - 31
        if K <= 0:
            continue
            
        # a <= b implies 2/a >= K/(31M) => a <= 62M/K
        max_a = min(M, (62 * M) // K)
        
        for a in range(1, max_a + 1):
            denom_b = a * K - 31 * M
            if denom_b <= 0:
                continue
                
            if (31 * M * a) % denom_b == 0:
                b = (31 * M * a) // denom_b
                
                if b < a:
                    continue
                if b > M:
                    continue
                
                # Check distinctness
                if a != b and a != M and b != M:
                    # Found a valid solution with largest denominator M
                    # Since M is increasing, this is the minimum possible M for 3 terms
                    return [a, b, M]
                    
    # If no 3-term solution with M < 248 is found, return the known best
    return best_solution
# EVOLVE-BLOCK-END
