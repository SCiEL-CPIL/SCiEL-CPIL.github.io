import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety
    
    N = 18
    MIN_DIST = 6
    
    # Try to construct QR(18) code: [18, 9, 6] with 512 words
    # Compute generator polynomial in F_2[x]
    
    # x^18 - 1 = (x+1)^2 * (x^8+x^7+...+x+1)^2 in F_2[x]
    # We need to find the QR generator polynomial
    
    # Let's compute the minimal polynomials
    # Cyclotomic cosets mod 19 under mult by 2:
    # Since ord_19(2) = 18, there's one coset {1,2,4,8,16,13,7,14,9,18,17,15,11,3,6,12,5,10}
    # and coset {0}
    
    # This means x^18-1 = (x+1) * M(x) where M is irreducible of degree 18
    # But wait, that contradicts the factorization above. Let me recheck.
    
    # Actually, x^18 - 1 in F_2[x]:
    # The roots are all 18th roots of unity in the algebraic closure of F_2.
    # An element α with α^18 = 1 lies in GF(2^d) where d | 18 and 18 | (2^d - 1).
    # 2^1 - 1 = 1, 2^2 - 1 = 3, 2^3 - 1 = 7, 2^6 - 1 = 63, 2^9 - 1 = 511, 2^18 - 1 = 262143
    # 18 | 63? No. 18 | 511? 511/18 = 28.39, no. 18 | 262143? 262143/18 = 14563.5, no.
    # Wait, 2^18 - 1 = 262143. 262143 / 18 = 14563.5. Not divisible!
    
    # I made an error. Let me recompute. The order of 2 mod 19 is 18, meaning 2^18 ≡ 1 (mod 19).
    # This means a primitive 19th root of unity has order 19, not 18.
    # For the QR code of length n = p-1 = 18, we need 18th roots of unity.
    # An 18th root of unity α satisfies α^18 = 1, so α lies in GF(2^d) where 18 | (2^d - 1).
    # 2^1-1=1, 2^2-1=3, 2^3-1=7, 2^6-1=63 (18|63? 63/18=3.5, no), 
    # 2^9-1=511 (18|511? 511/18≈28.4, no), 2^18-1=262143 (18|262143? 262143/18=14563.5, no)
    
    # Hmm, 18 does not divide 2^d - 1 for any d? That can't be right.
    # 2^d ≡ 1 (mod 18)? 18 = 2*9. We need 2^d ≡ 1 (mod 2) and 2^d ≡ 1 (mod 9).
    # 2^d ≡ 1 (mod 2) is always true for d ≥ 1.
    # 2^d ≡ 1 (mod 9): order of 2 mod 9 is 6 (2^1=2, 2^2=4, 2^3=8, 2^4=16≡7, 2^5=14≡5, 2^6=10≡1).
    # So we need d ≡ 0 (mod 6). Smallest d = 6.
    # Check: 18 | (2^6 - 1) = 63? 63/18 = 3.5. No!
    
    # I'm confusing myself. The condition for a primitive n-th root of unity to exist in GF(2^d) 
    # is that n | (2^d - 1). For n = 18: 18 | (2^d - 1).
    # 2^d mod 18: d=1:2, d=2:4, d=3:8, d=4:16, d=5:14, d=6:10, d=7:2, ... cycle of length 6
    # 2^d is never 1 mod 18 for d ≥ 1 (since 2^d is even, can't be ≡ 1 mod 2... wait 1 is odd, 2^d is even, so 2^d - 1 is odd, can't be divisible by 2, hence not by 18).
    
    # So there are NO 18th roots of unity in any GF(2^d)! This means x^18 - 1 has no linear factors 
    # other than possibly x+1 (for the root 1).
    
    # x^18 - 1 = (x-1)(x^17 + x^16 + ... + x + 1) in any field. In F_2, x-1 = x+1.
    # x^17 + x^16 + ... + x + 1: does this factor?
    # Since 18 doesn't divide 2^d-1 for any d, this polynomial is... 
    # Actually, the factorization of x^n - 1 in F_q[x] depends on the cyclotomic cosets.
    
    # For n=18, the cyclotomic cosets mod 18 under multiplication by 2:
    # C_0 = {0}
    # C_1 = {1, 2, 4, 8, 16, 14, 10, 2} wait: 1*2=2, 2*2=4, 4*2=8, 8*2=16, 16*2=32≡14, 14*2=28≡10, 10*2=20≡2... 
    # So C_1 = {1,2,4,8,16,14,10} (size 7? let me recheck)
    # 1→2→4→8→16→32mod18=14→28mod18=10→20mod18=2. So C_1={1,2,4,8,16,14,10}, size 7.
    # C_3 = {3,6,12} (3→6→12→24mod18=6), size 3.
    # C_5 = {5,10,2} wait 5→10→20mod18=2→4→8→16→14→10. So C_5={5,10,2,4,8,16,14}, size 7. Same as C_1? 
    # 2 is in C_1, so C_5 = C_1.
    # C_7 = {7,14,10,2,4,8,16} = C_1.
    # C_9 = {9} (9→18mod18=0... wait, we're working mod 18, and 9*2=18≡0). So C_9={9,0}={0,9}.
    # C_11 = {11,4,8,16,14,10,2} = C_1.
    # C_13 = {13,8,16,14,10,2,4} = C_1.
    # C_15 = {15,12,6,3} = C_3 ∪ {15}. 15→30mod18=12→24mod18=6→12... So C_15={15,12,6}, and C_3={3,6,12}. 
    # 3→6→12→24mod18=6. So C_3={3,6,12}. 15→30mod18=12→24mod18=6→12. So C_15={15,12,6}. 
    # These are different! C_3={3,6,12}, C_15={15,12,6}. But 12 is in both. 
    # Wait, 15*2=30, 30 mod 18=12. 12*2=24, 24 mod 18=6. 6*2=12. So C_15={15,12,6}.
    # And C_3: 3*2=6, 6*2=12, 12*2=24mod18=6. So C_3={3,6,12}.
    # 12 is in both C_3 and C_15, which is impossible for a partition. Let me recheck.
    # 12*2 = 24, 24 mod 18 = 6. 6*2=12. So from 12 we get {12,6}. From 3: 3→6→12→6. So C_3={3,6,12}.
    # From 15: 15→30mod18=12→24mod18=6→12. So C_15={15,12,6}. But 12 and 6 are already in C_3!
    # This means my coset computation is wrong. The cosets should partition {0,...,17}.
    
    # Let me redo carefully. Cosets of <2> in Z/18Z:
    # Start with 0: {0}
    # Start with 1: 1,2,4,8,16,32mod18=14,28mod18=10,20mod18=2. So {1,2,4,8,16,14,10}
    # Start with 3: 3,6,12,24mod18=6. So {3,6,12}
    # Start with 5: 5,10,20mod18=2,4,8,16,14,10. Already in C_1. So 5 is in C_1? 
    # 5*2=10, 10 is in C_1. So yes, 5 ∈ C_1. But I listed C_1 as {1,2,4,8,16,14,10} which doesn't include 5!
    # Let me recompute: 1→2→4→8→16→32mod18=14→28mod18=10→20mod18=2. 
    # So the coset is {1,2,4,8,16,14,10}. 5 is NOT in this set. But 5*2=10 which IS in the set. 
    # Contradiction! Unless 5 is in a different coset that maps to C_1.
    # No, if 5*2=10 and 10 ∈ C_1, then 5 must be in C_1 (since cosets are closed under *2).
    # So I must have made an error. Let me recompute 16*2: 16*2=32, 32-18=14. 14*2=28, 28-18=10. 10*2=20, 20-18=2.
    # So C_1 = {1,2,4,8,16,14,10}. Now 5*2=10 ∈ C_1, so 5 should be in C_1. But 5 is not in the set I computed.
    # This means the coset is larger. Let me start from 5: 5→10→2→4→8→16→14→10. 
    # So {5,10,2,4,8,16,14}. This is the same set as C_1 plus 5! 
    # So C_1 = {1,2,4,5,8,10,14,16}? Let me check: 1→2→4→8→16→14→10→2. 
    # Where does 5 come in? 5 is not reached from 1 by repeated *2. But 5*2=10 which is reached.
    # Ah, I see the issue. The coset of 1 is {1,2,4,8,16,14,10} (7 elements). The coset of 5 is {5,10,2,4,8,16,14} (7 elements). 
    # These overlap! That's impossible. Unless I'm computing wrong.
    
    # 5*2 = 10. 10*2 = 20 mod 18 = 2. 2*2=4. 4*2=8. 8*2=16. 16*2=32 mod 18=14. 14*2=28 mod 18=10.
    # So from 5: {5,10,2,4,8,16,14}. From 1: {1,2,4,8,16,14,10}.
    # The intersection is {2,4,8,10,14,16}. This is clearly wrong for a group action.
    
    # I think the issue is that the multiplicative group (Z/18Z, ×) is not a group since not all elements are coprime to 18.
    # The cyclotomic cosets are defined for the action of q on Z/nZ by multiplication, but this is only a permutation if gcd(q,n)=1.
    # gcd(2,18)=2 ≠ 1, so multiplication by 2 is NOT a permutation of Z/18Z!
    
    # This is the key issue. For n=18 and q=2, since gcd(2,18)≠1, the standard cyclotomic coset decomposition doesn't apply directly.
    # x^18 - 1 in F_2[x] needs a different factorization approach.
    
    # x^18 - 1 = (x^9)^2 - 1 = (x^9 - 1)^2 in F_2[x] (since char 2, (a-b)^2 = a^2 - 2ab + b^2 = a^2 + b^2)
    # Wait: (x^9 + 1)^2 = x^18 + 2x^9 + 1 = x^18 + 1 in F_2. And x^18 - 1 = x^18 + 1 in F_2.
    # So x^18 + 1 = (x^9 + 1)^2 in F_2[x].
    # x^9 + 1 = (x+1)(x^8+x^7+x^6+x^5+x^4+x^3+x^2+x+1) in F_2[x].
    # So x^18+1 = (x+1)^2(x^8+x^7+...+x+1)^2.
    
    # Now x^8+x^7+...+x+1 = (x^9-1)/(x-1) = (x^9+1)/(x+1) in F_2[x].
    # Does x^8+x^7+...+x+1 factor further? Its roots are the 9th roots of unity (excluding 1).
    # 9th roots of unity: α^9=1. Need 9 | (2^d - 1). 2^6=64, 64 mod 9 = 1. So order of 2 mod 9 is 6.
    # So 9th roots of unity lie in GF(2^6). The cyclotomic cosets mod 9 under *2:
    # C_0={0}, C_1={1,2,4,8,7,5} (1→2→4→8→16mod9=7→14mod9=5→10mod9=1), C_3={3,6} (3→6→12mod9=3).
    # So x^8+x^7+...+x+1 = M_1(x) * M_3(x) where M_1 has degree 6 and M_3 has degree 2.
    
    # M_3(x) = (x - α^3)(x - α^6) where α^9=1. Since α^6 = α^(-3), and in F_2, M_3 = x^2 + x + 1 (the only irreducible of degree 2 over F_2 with 9th roots).
    # Actually, α^3 has order 3, so it's a root of x^2+x+1. Yes, M_3 = x^2+x+1.
    # M_1(x) has degree 6 and is irreducible.
    
    # So x^18+1 = (x+1)^2 * (x^2+x+1)^2 * M_1(x)^2 where M_1 is irreducible of degree 6.
    
    # The QR(18) code generator polynomial should be one of the factors. For a [18,9,6] code, 
    # the generator polynomial should have degree 9.
    # Possible degree-9 factors: (x+1)^2 * M_1(x) has degree 2+6=8. (x+1)*M_1(x)^2 has degree 1+12=13. 
    # (x+1)^2*(x^2+x+1)*M_1(x) has degree 2+2+6=10. Hmm.
    # (x+1)*(x^2+x+1)^2*M_1(x) has degree 1+4+6=11.
    # (x^2+x+1)^2*M_1(x) has degree 4+6=10.
    # (x+1)^2*(x^2+x+1)^2 has degree 2+4=6.
    
    # None of these give degree 9. So QR(18) might not be a cyclic code in the standard sense, 
    # or I'm making an error.
    
    # Let me take a completely different approach: use a greedy/local search algorithm.
    
    # Start with the Reed-Muller code (32 words) and try to add more words.
    
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    # Reed-Muller RM(1,4) extended to length 18
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    words_set = set(words)
    
    # Greedy: try to add words with minimum distance >= 6 from all existing words
    # Use a time-limited search
    import random
    random.seed(42)
    
    added = 0
    attempts = 0
    max_attempts = 20000000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # Generate random candidate
        if attempts % 10000 == 0:
            # Try structured candidates
            for slope in range(16):
                for intercept in range(2):
                    word = 0
                    for point in range(16):
                        bit = ((slope & point).bit_count() + intercept) % 2
                        word |= bit << point
                    # Add some random bits in positions 16,17
                    word |= (random.randint(0,1) << 16) | (random.randint(0,1) << 17)
                    if word not in words_set:
                        ok = True
                        for w in words:
                            if hamming_dist(word, w) < 6:
                                ok = False
                                break
                        if ok:
                            words.append(word)
                            words_set.add(word)
                            added += 1
        else:
            word = random.randint(0, (1 << 18) - 1)
            if word in words_set:
                continue
            ok = True
            for w in words:
                if hamming_dist(word, w) < 6:
                    ok = False
                    break
            if ok:
                words.append(word)
                words_set.add(word)
                added += 1
    
    return words
