import time
from math import gcd

N = 512

def _divisors_of(n):
    """Return all divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def _count_divisors(n):
    """Count the number of divisors of n."""
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def _is_valid_set(S, N):
    """Check if all pairs in S have lcm <= N."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a // gcd(a, b)) * b > N:
                return False
    return True

def construct():
    # Find L in [1, 512] that maximizes the number of divisors
    max_divs = 0
    best_L = 1
    
    for L in range(1, N + 1):
        d_count = _count_divisors(L)
        if d_count > max_divs:
            max_divs = d_count
            best_L = L
            
    # The set of divisors of best_L is a valid set
    # Because for any a, b in D(best_L), lcm(a, b) divides best_L, so lcm(a, b) <= best_L <= 512
    result = _divisors_of(best_L)
    
    # Optional: Verify validity (should always be true by construction)
    if not _is_valid_set(result, N):
        # Fallback to a known valid set, e.g., divisors of 1
        return [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                valid = False
                break
        if not valid:
            break
    if valid:
        print("All pairs valid.")
