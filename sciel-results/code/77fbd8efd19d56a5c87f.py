"""Solve: find maximum cardinality subset of [1,512] with pairwise lcm <= 512."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 80
    n = 512
    limit = 512
    
    # Build incompatibility: incompatible if lcm(a,b) > 512
    # Use 0-indexed: element i represents value i+1
    incompat = [0] * n
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g = gcd(a, b)
            if a * b // g > limit:
                incompat[a-1] |= (1 << b)
                incompat[b-1] |= (1 << a)
    
    # Degrees
    degrees = [bin(incompat[i]).count('1') for i in range(n)]
    
    # Order by degree ascending (include low-degree first)
    order = list(range(n))
    order.sort(key=lambda i: degrees[i])
    
    # Greedy construction
    current = 0  # bitset
    included = []
    for idx in order:
        if (incompat[idx] & current) == 0:
            current |= (1 << idx)
            included.append(idx)
    
    best_size = len(included)
    best_list = list(included)
    
    # Local search: try to replace elements to increase size
    # Strategy: for each element, try removing it and see if we can add more
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_list = list(best_list)
        for i in range(len(cur_list)):
            if time.time() > deadline:
                break
            removed = cur_list[i]
            temp = [x for x in cur_list if x != removed]
            temp_bitset = 0
            for x in temp:
                temp_bitset |= (1 << x)
            
            # Try to find an element j not in temp that is compatible with all in temp
            # and that would give us a larger set
            for j in range(n):
                if time.time() > deadline:
                    break
                if (temp_bitset >> j) & 1:
                    continue
                if (incompat[j] & temp_bitset) == 0:
                    # j is compatible with all in temp
                    if len(temp) + 1 > len(cur_list):
                        cur_list = temp + [j]
                        improved = True
                        if len(cur_list) > best_size:
                            best_size = len(cur_list)
                            best_list = list(cur_list)
                        break
            if improved:
                break
    
    # Also try: start from empty, greedy in different orders
    # Try ordering by value
    order2 = list(range(n))
    current2 = 0
    included2 = []
    for idx in order2:
        if (incompat[idx] & current2) == 0:
            current2 |= (1 << idx)
            included2.append(idx)
    if len(included2) > best_size:
        best_size = len(included2)
        best_list = list(included2)
    
    # Try ordering by value descending
    order3 = list(range(n - 1, -1, -1))
    current3 = 0
    included3 = []
    for idx in order3:
        if (incompat[idx] & current3) == 0:
            current3 |= (1 << idx)
            included3.append(idx)
    if len(included3) > best_size:
        best_size = len(included3)
        best_list = list(included3)
    
    result_values = sorted([x + 1 for x in best_list])
    return result_values
# EVOLVE-BLOCK-END
