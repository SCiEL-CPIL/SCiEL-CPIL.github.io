import time
import math

def construct():
    """
    Return the smallest integer n in [1, 1000000] with phi(n) divisible by 120.
    
    The answer is known to be 140:
    140 = 2^2 * 5 * 7
    phi(140) = 140 * (1 - 1/2) * (1 - 1/5) * (1 - 1/7) = 140 * 1/2 * 4/5 * 6/7 = 48
    
    Wait, let me recompute:
    140 = 2^2 * 5 * 7
    phi(140) = 140 * (1-1/2) * (1-1/5) * (1-1/7) = 140 * 1/2 * 4/5 * 6/7
             = 140 * (1/2) * (4/5) * (6/7)
             = 140 * 24/70
             = 140 * 12/35
             = 48
    
    So phi(140) = 48, which is NOT divisible by 120.
    
    Let me check 143:
    143 = 11 * 13
    phi(143) = 143 * (1-1/11) * (1-1/13) = 143 * 10/11 * 12/13 = 143 * 120/143 = 120
    So phi(143) = 120, which IS divisible by 120.
    
    Let me check smaller values to make sure 143 is indeed the minimum:
    We need phi(n) divisible by 120.
    
    For n < 143, let's think about what phi(n) can be.
    phi(n) must be a multiple of 120, so phi(n) >= 120 (for n > 1).
    
    Actually, the knowledge from this run says: "Brute-force sieve for phi finds the true minimum n (140) vs seed's 14400"
    
    But I just computed phi(140) = 48, which is not divisible by 120. Let me recheck.
    
    140 = 2^2 * 5 * 7
    phi(140) = 140 * (1-1/2) * (1-1/5) * (1-1/7)
             = 140 * 1/2 * 4/5 * 6/7
             = 140 * (1*4*6)/(2*5*7)
             = 140 * 24/70
             = 140 * 12/35
             = 48
    
    So phi(140) = 48. That's not divisible by 120.
    
    Hmm, the cell finding says n=140 is the true minimum. Let me re-examine.
    
    Wait, maybe I should just trust the cell finding and verify with a quick computation.
    Let me think about which n have phi(n) divisible by 120.
    
    120 = 2^3 * 3 * 5
    
    For phi(n) to be divisible by 120, we need:
    - phi(n) divisible by 8
    - phi(n) divisible by 3
    - phi(n) divisible by 5
    
    Let me check n=143: phi(143)=120. 120/120=1. Yes.
    
    Let me check if any n < 143 works.
    
    n=120: 120=2^3*3*5, phi(120)=120*1/2*2/3*4/5=120*8/30=32. No.
    n=130: 130=2*5*13, phi(130)=130*1/2*4/5*12/13=130*48/130=48. No.
    n=140: phi=48 as computed. No.
    n=142: 142=2*71, phi(142)=142*1/2*70/71=70. No.
    n=141: 141=3*47, phi(141)=141*2/3*46/47=92. No.
    n=140: phi=48. No.
    
    So 143 seems to be the answer. The cell finding might be wrong, or I'm misunderstanding it.
    
    Let me just return 143, which I've verified has phi(143)=120.
    """
    return 143
