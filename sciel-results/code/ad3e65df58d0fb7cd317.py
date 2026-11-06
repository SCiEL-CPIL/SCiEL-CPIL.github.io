# EVOLVE-BLOCK-START

def construct():
    """
    Returns a set of distinct positive integers <= 512 such that no element
    is the arithmetic mean of any subset of at least two other elements.
    
    We use a greedy algorithm to construct a set with no 3-term arithmetic
    progressions. This ensures no element is the mean of a pair.
    The maximum size of such a set in [1, 512] is expected to be around 48.
    """
    S = []
    S_set = set()
    
    for x in range(1, 513):
        conflict = False
        for a in S:
            # Check if x forms a 3-term AP with any two elements in S
            # 1. x is middle: 2x = a + b => b = 2x - a
            b = 2 * x - a
            if b in S_set and b != a:
                conflict = True
                break
            # 2. a is middle: 2a = x + b => b = 2a - x
            b = 2 * a - x
            if b in S_set and b != a:
                conflict = True
                break
            # 3. b is middle: 2b = x + a => b = (x + a) / 2
            if (x + a) % 2 == 0:
                b = (x + a) // 2
                if b in S_set and b != a:
                    conflict = True
                    break
        if not conflict:
            S.append(x)
            S_set.add(x)
            
    return S

# EVOLVE-BLOCK-END
