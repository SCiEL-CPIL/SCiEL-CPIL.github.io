"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Start with all odd numbers in [1, 256]
    # Odds are 1, 3, 5, ..., 255. There are 128 odds.
    # For any two distinct odds a, b:
    # a+b is even. a*b is odd.
    # (a*b) % (a+b) != 0 because a+b is even and a*b is odd,
    # so a+b cannot divide a*b (an even number cannot divide an odd number).
    # Thus, all odds are mutually compatible.
    
    S = set(range(1, 257, 2))  # All odd numbers
    
    # Greedily add even numbers that are compatible with all current members
    # For an even number e and an odd number o:
    # e + o is odd. e * o is even.
    # (e * o) % (e + o) == 0 iff (e + o) divides (e * o).
    # Let d = e + o (odd). Then o = d - e.
    # d | e * o = e * (d - e) = e * d - e^2
    # So d | e * o iff d | e^2.
    # Since d is odd, d | e^2 iff d | (e/2^v2(e))^2 where v2(e) is the 2-adic valuation.
    # More simply: for each even e, check if it's compatible with all current S.
    
    # We'll try to add even numbers from 256 down to 2 (larger first, as they have fewer small divisors)
    # Actually, let's just iterate through all evens and check compatibility.
    # To be deterministic and potentially better, let's try largest first.
    
    evens = list(range(256, 0, -2))
    
    for e in evens:
        # Check compatibility with all current members
        compatible = True
        for m in S:
            if (e * m) % (e + m) == 0:
                compatible = False
                break
        if compatible:
            S.add(e)
    
    return list(S)
# EVOLVE-BLOCK-END
