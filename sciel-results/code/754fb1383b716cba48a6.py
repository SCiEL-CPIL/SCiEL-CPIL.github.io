def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    We are looking for the minimum n such that 120 | phi(n).
    120 = 2^3 * 3 * 5.
    
    Checking small values:
    - Primes p: phi(p) = p-1. We need p = 120k + 1.
      Smallest prime p = 241 (since 121=11^2, 241 is prime). phi(241)=240, div by 120.
    - Prime powers p^k: 
      If p=2, phi(2^k)=2^(k-1), no factor 3 or 5.
      If p=3, phi(3^k)=2*3^(k-1), no factor 5.
      If p=5, phi(5^k)=4*5^(k-1), no factor 3.
      Other primes p, phi(p^k) has factor p-1. If p-1 is not div by 120, need higher powers?
      e.g., p=7, phi(7^k)=6*7^(k-1). No factor 5.
      p=11, phi(11^k)=10*11^(k-1). No factor 3.
      p=13, phi(13^k)=12*13^(k-1). No factor 5.
      p=19, phi(19^k)=18*19^(k-1). No factor 5.
      p=23, phi(23^k)=22*23^(k-1). No factor 3, 5.
      p=31, phi(31^k)=30*31^(k-1). No factor 2^3.
      p=43, phi(43^k)=42*43^(k-1). No factor 5.
      p=61, phi(61^k)=60*61^(k-1). No factor 2^3.
      p=67, phi(67^k)=66*67^(k-1). No factor 5.
      p=109, phi(109^k)=108*109^(k-1). No factor 5.
      p=181, phi(181^k)=180*181^(k-1). 180 = 2^2 * 3^2 * 5. Missing 2.
      p=241 is prime, handled above.
      
    - Composites n = p^a q^b ...
      phi(n) = product p_i^(a_i-1) (p_i-1).
      We need the product to be divisible by 2^3 * 3 * 5.
      
      Let's check n < 143.
      If n is square-free, n = p q. phi(n) = (p-1)(q-1).
      We need (p-1)(q-1) div by 120.
      Possible pairs (p,q) with p<q, p*q < 143:
      (2,3): phi=2. No.
      (2,5): phi=4. No.
      (2,7): phi=6. No.
      (2,11): phi=10. No.
      (2,13): phi=12. No.
      (2,17): phi=16. No.
      (2,19): phi=18. No.
      (2,23): phi=22. No.
      (2,29): phi=28. No.
      (2,31): phi=30. No.
      (2,37): phi=36. No.
      (2,41): phi=40. No.
      (2,43): phi=42. No.
      (2,47): phi=46. No.
      (2,53): phi=52. No.
      (2,59): phi=58. No.
      (2,61): phi=60. No.
      (2,67): phi=66. No.
      (2,71): phi=70. No.
      (2,73): phi=72. No.
      (2,79): phi=78. No.
      (2,83): phi=82. No.
      (2,89): phi=88. No.
      (2,97): phi=96. No.
      (2,101): phi=100. No.
      (2,103): phi=102. No.
      (2,107): phi=106. No.
      (2,109): phi=108. No.
      (2,113): phi=112. No.
      (2,127): phi=126. No.
      (3,5): phi=8. No.
      (3,7): phi=12. No.
      (3,11): phi=20. No.
      (3,13): phi=24. No.
      (3,17): phi=32. No.
      (3,19): phi=36. No.
      (3,23): phi=42. No.
      (3,29): phi=56. No.
      (3,31): phi=60. No.
      (3,37): phi=72. No.
      (3,41): phi=80. No.
      (3,43): phi=84. No.
      (3,47): phi=92. No.
      (3,53): phi=104. No.
      (3,59): phi=116. No.
      (3,61): phi=120. YES. n=3*61=183. > 143.
      (5,7): phi=24. No.
      (5,11): phi=40. No.
      (5,13): phi=48. No.
      (5,17): phi=64. No.
      (5,19): phi=72. No.
      (5,23): phi=88. No.
      (5,29): phi=112. No.
      (5,31): phi=120. YES. n=5*31=155. > 143.
      (7,11): phi=60. No.
      (7,13): phi=72. No.
      (7,17): phi=96. No.
      (7,19): phi=108. No.
      (7,23): phi=120. YES. n=7*23=161. > 143.
      (11,13): phi=120. YES. n=11*13=143. < 155, 161, 183.
      
      So 143 is the smallest square-free composite solution.
      
      What about non-square-free n < 143?
      n = p^2 q ...
      If n < 143, p must be small.
      p=2: n=4q. phi(4q)=2*phi(q) if q odd prime.
      We need 120 | 2*phi(q) => 60 | phi(q).
      Smallest prime q with phi(q)=q-1 div by 60? q=61. n=4*61=244. > 143.
      If q composite, n=4*q.
      p=3: n=9q. phi(9q)=6*phi(q).
      We need 120 | 6*phi(q) => 20 | phi(q).
      Smallest prime q with q-1 div by 20? q=21 (no), 41. n=9*41=369. > 143.
      p=5: n=25q. phi(25q)=20*phi(q).
      We need 120 | 20*phi(q) => 6 | phi(q).
      Smallest prime q with q-1 div by 6? q=7. n=25*7=175. > 143.
      p=7: n=49q. phi(49q)=42*phi(q).
      We need 120 | 42*phi(q) => 60/18? No.
      120/42 = 20/7. Not integer.
      gcd(120, 42) = 6.
      120/6 = 20. 42/6 = 7.
      We need 20 | 7*phi(q). Since gcd(20,7)=1, 20 | phi(q).
      Smallest prime q with q-1 div by 20? q=41. n=49*41=2009. > 143.
      
      It seems 143 is the global minimum.
    """
    return 143
