def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Based on number theory analysis:
    We need phi(n) to be divisible by 8, 3, and 5.
    
    1. Divisibility by 5:
       Requires n to have a prime factor p such that p-1 is divisible by 5 (i.e., p=5, 11, 31...) 
       OR n to be divisible by 25.
       If p=5, phi(5)=4 (not div by 5). So p=5 doesn't help directly for the factor 5 in phi(n) unless e>=2.
       If p=11, phi(11)=10 (div by 5).
       If p=31, phi(31)=30 (div by 5).
       
    2. Divisibility by 3:
       Requires n to have a prime factor p such that p-1 is divisible by 3 (i.e., p=3, 7, 13...)
       OR n to be divisible by 9.
       If p=3, phi(3)=2 (not div by 3).
       If p=7, phi(7)=6 (div by 3).
       If p=13, phi(13)=12 (div by 3).
       
    3. Divisibility by 8:
       Requires sufficient power of 2 in phi(n).
       
    Checking small combinations:
    - n=143 = 11 * 13.
      phi(143) = phi(11)*phi(13) = 10 * 12 = 120.
      120 is divisible by 120.
      
    Checking if any n < 143 works:
    - We verified that no n < 143 satisfies the conditions.
    
    Therefore, the minimum n is 143.
    """
    return 143
