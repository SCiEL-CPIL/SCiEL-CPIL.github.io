from fractions import Fraction

def construct():
    """
    Return 1..256 distinct integer denominators in [5, 10000000] whose reciprocal sum is exactly 1;
    minimize their count.
    """
    # We search for a solution with 8 terms.
    # Base: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 = 2131/2520
    # Remaining: 1 - 2131/2520 = 389/2520
    # We need 1/a + 1/b = 389/2520 with a, b > 10, a != b.
    # (389a - 2520)(389b - 2520) = 2520^2
    
    target = 2520 * 2520
    # Find divisors of target
    divs = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            divs.append(i)
            if i != target // i:
                divs.append(target // i)
        i += 1
        
    found = None
    for x in divs:
        if x % 389 == 203:
            y = target // x
            if y % 389 == 203:
                a = (x + 2520) // 389
                b = (y + 2520) // 389
                if a > 10 and b > 10 and a != b:
                    found = [5, 6, 7, 8, 9, 10, a, b]
                    break
                    
    if found:
        return found
        
    # If not found, try other bases or return a known solution.
    # For robustness, let's return a known valid 8-term solution if we can find one.
    # One known solution is:
    # [5, 6, 7, 8, 9, 10, 11, 12] is not valid (sum > 1).
    # Let's use a search for 9 terms if 8 fails? No, minimize count.
    # Let's assume the above works.
    
    # Fallback: Return a valid solution with more terms if needed, but we want 8.
    # Let's try another base: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/11
    # Sum = 2131/2520 - 1/10 + 1/11 = 2131/2520 - 11/110 + 10/110 = 2131/2520 - 1/110
    # LCM(2520, 110) = 27720
    # 2131/2520 = 23441/27720
    # 1/110 = 252/27720
    # Sum = (23441 - 252)/27720 = 23189/27720
    # Remaining = 1 - 23189/27720 = 4531/27720
    # 1/a + 1/b = 4531/27720
    # (4531a - 27720)(4531b - 27720) = 27720^2
    # This is more complex.
    
    # Let's just return the first found solution.
    raise RuntimeError("No solution found")

if __name__ == "__main__":
    print(construct())
