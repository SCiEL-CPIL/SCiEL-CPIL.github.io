"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # We need the smallest n such that phi(n) % 120 == 0.
    # Let's verify the small candidates manually or with a very fast check.
    # phi(1) = 1
    # phi(2) = 1
    # phi(3) = 2
    # phi(4) = 2
    # phi(5) = 4
    # phi(6) = 2
    # phi(7) = 6
    # phi(8) = 4
    # phi(9) = 6
    # phi(10) = 4
    # phi(11) = 10
    # phi(12) = 4
    # phi(13) = 12
    # phi(14) = 6
    # phi(15) = 8
    # phi(16) = 8
    # phi(17) = 16
    # phi(18) = 6
    # phi(19) = 18
    # phi(20) = 8
    # ...
    # phi(n) is even for n > 2.
    # 120 = 2^3 * 3 * 5.
    # So phi(n) must be divisible by 8, 3, and 5.
    # This implies phi(n) >= 120 (unless phi(n)=0 which is impossible).
    # The smallest n such that phi(n) >= 120 is roughly 143? 
    # Let's check n around 100-200.
    # phi(143) = phi(11*13) = 10*12 = 120. 120 % 120 == 0.
    # So n=143 is a candidate.
    # Is there a smaller one?
    # We need phi(n) to be a multiple of 120. The smallest positive multiple is 120.
    # So we are looking for the smallest n with phi(n) = 120, 240, 360, etc.
    # Let's check if any n < 143 has phi(n) = 120.
    # Known values:
    # phi(120) = 32
    # phi(121) = 110
    # phi(122) = 60
    # phi(123) = 80
    # phi(124) = 60
    # phi(125) = 100
    # phi(126) = 36
    # phi(127) = 126
    # phi(128) = 64
    # phi(129) = 84
    # phi(130) = 48
    # phi(131) = 130
    # phi(132) = 40
    # phi(133) = 84
    # phi(134) = 66
    # phi(135) = 72
    # phi(136) = 64
    # phi(137) = 136
    # phi(138) = 44
    # phi(139) = 138
    # phi(140) = 48
    # phi(141) = 46
    # phi(142) = 70
    # phi(143) = 120.
    
    # So 143 is the smallest n with phi(n)=120.
    # Could there be a smaller n with phi(n) = 240? No, because phi(n) grows roughly with n.
    # If n < 143, phi(n) < n < 143. So phi(n) cannot be 240 or more.
    # Thus, the only multiple of 120 that phi(n) can be for n < 143 is 120.
    # And we checked n=1..142 (implicitly above, but let's be sure).
    # Actually, we just need to confirm no n < 143 has phi(n)=120.
    # Since phi(n) < n, if n < 143, phi(n) <= 142.
    # The only multiple of 120 in [1, 142] is 120.
    # So we just need the smallest n with phi(n)=120.
    # We found n=143.
    # Are there any n < 143 with phi(n)=120?
    # Let's check a few likely candidates for high phi(n) relative to n.
    # n=111: phi=72
    # n=115: phi=88
    # n=119: phi=96
    # n=121: phi=110
    # n=123: phi=80
    # n=125: phi=100
    # n=127: phi=126
    # n=129: phi=84
    # n=131: phi=130
    # n=133: phi=84
    # n=135: phi=72
    # n=137: phi=136
    # n=139: phi=138
    # n=141: phi=46
    # n=142: phi=70
    # n=143: phi=120.
    
    # It seems 143 is indeed the smallest.
    # To be absolutely safe and fast, we can just return 143.
    # The sieve in the original code is overkill and slower than a direct return.
    # However, the task says "Minimize n". If 143 is the answer, returning it directly is optimal.
    
    # Let's do a quick sanity check with a small loop just in case my manual derivation missed something.
    # But 143 is a very strong candidate.
    
    # To be robust against any edge cases I might have miscalculated, 
    # I will perform a very fast check for n up to 200 using a simple totient function.
    
    def phi_small(n):
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    for n in range(1, 200):
        if phi_small(n) % 120 == 0:
            return n
            
    # If somehow no n < 200 works (should not happen), fall back to a wider search or return 143.
    # But 143 works.
    return 143
# EVOLVE-BLOCK-END
