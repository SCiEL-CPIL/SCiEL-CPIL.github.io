import math

def construct():
    # Calculate LCM of 1 to 22
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    D = 1
    for i in range(1, 23):
        D = lcm(D, i)
        
    current_sums = {0}
    current_set = []
    
    # Try adding numbers from largest to smallest
    # This prioritizes "safe" elements (small reciprocals) first
    for x in range(22, 0, -1):
        inc = D // x
        # Generate new sums
        # Optimization: if current_sums is large, this set comprehension is O(2^k)
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + inc)
            
        # Check for collision
        if new_sums.isdisjoint(current_sums):
            current_sums.update(new_sums)
            current_set.append(x)
            
    return current_set
