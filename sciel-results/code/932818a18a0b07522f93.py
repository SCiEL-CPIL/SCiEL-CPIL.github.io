import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Primes 1 mod 4 for building M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421]
    
    # Small primes for Q (square base)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_N = 0
    best_pairs = []
    
    # Try different k values for Q = (product of first k small primes)^2
    # We want M to be as large as possible to have more prime factors, 
    # but Q must be a square of a product of primes to make N powerful? 
    # Wait, N = Q*M. For N to be powerful, every prime factor of N must appear with even exponent?
    # No, "powerful" means if p|n then p^2|n. 
    # If N = Q*M, and Q is a square, then any prime in Q has even exponent >= 2.
    # For M, if p|M, then p must divide M with exponent >= 2?
    # The task says: "a and b are powerful".
    # a = Q*x^2, b = Q*y^2.
    # If Q is a square, say Q = S^2, then a = S^2 x^2 = (Sx)^2. A perfect square is powerful.
    # So if we set Q to be a perfect square, then ANY x, y will result in a, b being powerful.
    # This simplifies things greatly. We just need N = Q*M where Q is a square, and M is any integer.
    # Then pairs are (Q*x^2, Q*y^2) for x^2+y^2=M.
    # To maximize pairs, we maximize the number of representations of M as sum of two squares.
    # Number of reps is maximized when M is a product of distinct primes p_i = 1 mod 4.
    # Let M = p1 * p2 * ... * pk.
    # Number of reps (x,y) with x>=1, x<=y is roughly 2^(k-1) or similar.
    
    # Strategy:
    # 1. Choose Q = S^2. S = product of small primes.
    # 2. Choose M = product of distinct primes p_i = 1 mod 4 such that Q*M <= 10^18.
    # 3. Generate all representations of M as sum of two squares using Gaussian integers.
    
    # We want to maximize the number of representations.
    # The number of representations r_2(M) = 4 * (d_1(M) - d_3(M)).
    # For M = prod p_i, r_2(M) = 4 * 2^k.
    # Number of pairs (a,b) with a<=b is r_2(M) / 4 (if no squares) or so.
    # Actually, each representation (x,y) gives a pair. 
    # If x!=y, (x,y) and (y,x) are distinct reps but same pair {a,b}.
    # If x=y, it's one rep.
    # We count distinct pairs [a,b] with a<=b.
    # If M is not a square, x!=y always (since 2x^2=M -> M=2x^2, but M is odd if all p_i are odd).
    # So pairs = r_2(M) / 4 = 2^(k-1).
    # We want to maximize k.
    
    # Maximize k such that (product of first k primes 1 mod 4) * Q <= 10^18.
    # Let's try different Q. Smaller Q allows larger M, thus more primes.
    # Minimal Q=1 (S=1). Then M <= 10^18.
    # Product of first 15 primes 1 mod 4:
    # 5*13*17*29*37*41*53*61*73*89*97*101*109*113*137
    # log10(5)=0.7, log10(13)=1.1, ...
    # Let's estimate.
    # 10^18 is 10^18.
    # 5*13=65
    # 65*17=1105
    # 1105*29=32045
    # 32045*37=1185665
    # 1185665*41=48612265
    # 48612265*53=2576450045
    # 2576450045*61=157163452745
    # 157163452745*73=11472932050385
    # 11472932050385*89=1021090952484265
    # 1021090952484265*97=99045822390973705
    # 99045822390973705*101=9993628061488344205
    # 9993628061488344205*109 > 10^18.
    # So with Q=1, we can fit 10 primes.
    # Pairs = 2^(10-1) = 512.
    
    # If we use Q > 1, M is smaller, fewer primes.
    # So Q=1 is best for maximizing count?
    # Wait, is 1 powerful? Yes.
    # Is N=1*M powerful? No, N doesn't need to be powerful. a and b need to be powerful.
    # a = 1*x^2 = x^2. Powerful.
    # b = 1*y^2 = y^2. Powerful.
    # So Q=1 is valid.
    
    # Why did the previous code use Q > 1? Maybe to ensure N is large? No, N just needs to be <= 10^18.
    # Or maybe to avoid trivialities? No.
    # Let's stick with Q=1 to maximize the number of prime factors in M.
    
    # However, we must ensure we can generate the pairs fast.
    # Generating reps for M with 10 primes involves 2^10 = 1024 combinations. Very fast.
    
    # Let's optimize the selection of primes.
    # We want to maximize the number of prime factors in M <= 10^18.
    # This is achieved by taking the smallest primes 1 mod 4.
    
    # Algorithm:
    # 1. Set Q = 1.
    # 2. Select largest set of distinct primes p_i = 1 mod 4 such that product P <= 10^18.
    # 3. M = P.
    # 4. Generate all representations of M as x^2 + y^2.
    # 5. Return pairs [[x^2, y^2], ...].
    
    # Let's verify the number of pairs.
    # If M = p1...pk, number of primitive reps is 2^(k-1).
    # Total reps r_2(M) = 4 * 2^k.
    # Pairs (x,y) with x>0, y>0, x<=y.
    # Each pair corresponds to 4 reps (x,y), (-x,y), (x,-y), etc?
    # No, r_2 counts ordered pairs (x,y) with signs.
    # If x!=0, y!=0, x!=y, each unordered pair {x,y} gives 8 reps? No.
    # (x,y), (y,x), (-x,y), ...
    # The number of pairs [a,b] with a<=b is r_2(M) / 8 if x!=y and x!=0.
    # If M is product of distinct primes 1 mod 4, M is not a square, so x!=y.
    # Also x!=0 because M is not a square (and not 2*square).
    # So number of pairs = 2^(k-2)?
    # Let's check k=1. M=5. 1^2+2^2=5. Pairs: [1,4]. Count=1.
    # Formula 2^(1-2) = 1/2? No.
    # r_2(5) = 8. (1,2), (2,1), (-1,2)...
    # Pairs: [1,4]. Just 1.
    # So pairs = 2^(k-1) / 2 ? No.
    # For k=1, pairs=1.
    # For k=2, M=5*13=65.
    # 1^2+8^2=65, 4^2+7^2=65.
    # Pairs: [1,64], [16,49]. Count=2.
    # 2^(2-1) = 2.
    # For k=3, M=5*13*17=1105.
    # Reps:
    # 1105 = 1^2+33^2 = 4^2+31^2 = 9^2+28^2 = 12^2+23^2.
    # Count=4.
    # 2^(3-1) = 4.
    # So number of pairs is 2^(k-1).
    
    # So we want to maximize k.
    # Max k such that product of first k primes 1 mod 4 <= 10^18.
    # We found k=10 fits.
    # Let's check k=11.
    # Product of first 10: ~10^19.
    # Wait, my previous estimate:
    # P_10 = 9993628061488344205 ~ 10^19.
    # This is > 10^18.
    # So k=9?
    # P_9 = 99045822390973705 ~ 10^17.
    # P_10 = P_9 * 101 ~ 10^19.
    # So k=9 is the max for Q=1.
    # Pairs = 2^(9-1) = 256.
    
    # Can we do better?
    # If we use Q > 1, we reduce M, so k decreases.
    # So Q=1 is optimal for count.
    
    # But wait, the previous score was 28.33.
    # 256 is much larger than 28.
    # Why did the previous code get 28?
    # Maybe it didn't find the optimal M?
    # Or maybe the brute force was too slow and it timed out, returning a small result?
    # The code had `max_iters = 15000000`.
    # If M ~ 10^17, sqrt(M) ~ 10^8.5.
    # Brute force would take forever.
    # So it likely fell back to a smaller M or timed out.
    
    # So the new algorithm should easily get ~256.
    
    # Let's implement the Gaussian integer multiplication to find reps.
    
    def sum_two_squares_reps(M):
        """
        Returns list of (x, y) with x>0, y>0, x<=y, x^2+y^2=M.
        Uses factorization of M into primes 1 mod 4.
        """
        if M < 2:
            return []
        
        # Factorize M into primes 1 mod 4.
        # Since we construct M from known primes, we can pass them.
        # But let's write a generic one for safety, or just use the list.
        
        # We will pass the list of primes used to construct M.
        pass

    # Better: Pass the list of primes.
    
    def generate_reps(primes):
        """
        Generates all representations of M = product(primes) as sum of two squares.
        primes: list of distinct primes p = 1 mod 4.
        """
        # For each p, find u,v such that u^2+v^2=p.
        # Since p is small (< 1000), we can brute force u.
        
        base_reps = []
        for p in primes:
            found = False
            for u in range(1, int(math.isqrt(p)) + 1):
                v2 = p - u*u
                v = math.isqrt(v2)
                if v*v == v2 and v > 0:
                    base_reps.append((u, v))
                    found = True
                    break
            if not found:
                # Should not happen for p=1 mod 4
                base_reps.append((1, 1)) # Fallback, incorrect but avoids crash
        
        # Start with (1, 0) representing 1.
        # We want to combine them.
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # We maintain a list of (x, y) representing x+iy.
        # Initially [(1, 0)].
        # For each base rep (u, v), multiply all existing reps by (u, v) and (-u, v)?
        # No, the representations of p are (u,v), (-u,v), (u,-v), (-u,-v) etc.
        # In Gaussian integers, p = (u+iv)(u-iv).
        # The factors are u+iv and u-iv.
        # To get all reps of M, we choose for each p one of the two factors (u+iv) or (u-iv).
        # There are 2^k choices.
        # Each choice gives a Gaussian integer z = prod (u_i + i v_i * s_i) where s_i in {1, -1}.
        # Then z = x + iy is a representation.
        
        reps = [(1, 0)]
        for (u, v) in base_reps:
            new_reps = []
            for (x, y) in reps:
                # Multiply (x+iy) by (u+iv)
                x1 = x*u - y*v
                y1 = x*v + y*u
                new_reps.append((x1, y1))
                
                # Multiply (x+iy) by (u-iv)
                x2 = x*u + y*v
                y2 = -x*v + y*u
                new_reps.append((x2, y2))
            reps = new_reps
        
        # Filter valid reps: x>0, y>0, x<=y
        pairs = []
        seen = set()
        for (x, y) in reps:
            if x < 0: x = -x
            if y < 0: y = -y
            if x == 0 or y == 0: continue
            if x > y:
                x, y = y, x
            if (x, y) not in seen:
                seen.add((x, y))
                pairs.append((x, y))
        
        return pairs

    # Main logic
    # 1. Determine max k
    # 2. Construct M
    # 3. Generate pairs
    # 4. Scale by Q=1
    
    # Let's try to maximize k.
    # We can also try to include some primes with higher powers?
    # If p^e | M, the number of factors is e+1.
    # But p^2 is larger than p*q for small q?
    # 5^2 = 25. 13 is 13.
    # Replacing 13 with 5^2?
    # M = 5^2 * 17 * ...
    # Log(25) = 1.4. Log(13) = 1.1.
    # So 5^2 is "more expensive" than 13.
    # But it adds 2 factors (e=2 -> 3 choices? No, e+1=3) vs 1 factor (e=1 -> 2 choices).
    # Wait, the number of Gaussian factors is e+1?
    # If M = p^e, the number of ways to split the factors (u+iv)^e is e+1?
    # No, the number of representations is e+1.
    # So we gain 1 rep per extra exponent, but pay in size.
    # Is it worth it?
    # Size ratio: p^2 / p_next.
    # If p_next > p, then p^2 > p * p_next? No.
    # 5^2 = 25. Next prime 13. 5*13=65.
    # 25 < 65.
    # So replacing 13 with 5^2 reduces M size, allowing more primes?
    # Or allows higher exponent of 5?
    # Let's stick to distinct primes first, as it's simpler and likely optimal for count given size constraint.
    # Actually, let's just use the smallest primes.
    
    M = 1
    used_primes = []
    for p in p_1mod4:
        if M * p > 10**18:
            break
        M *= p
        used_primes.append(p)
    
    N = M
    pairs_xy = generate_reps(used_primes)
    
    # Convert to [a, b]
    final_pairs = []
    for (x, y) in pairs_xy:
        a = x * x
        b = y * y
        if a <= b:
            final_pairs.append([a, b])
        else:
            final_pairs.append([b, a])
    
    # Remove duplicates just in case
    unique_pairs = []
    seen_pairs = set()
    for p in final_pairs:
        if tuple(p) not in seen_pairs:
            seen_pairs.add(tuple(p))
            unique_pairs.append(p)
            
    return {'N': N, 'pairs': unique_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
