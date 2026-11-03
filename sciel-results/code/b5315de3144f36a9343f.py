import itertools
import sys

def construct(n):
    if n == 1:
        return "1"
    
    # Generate permutations in Johnson-Trotter order
    # This order ensures that consecutive permutations differ by a swap of adjacent elements,
    # leading to higher overlap (n-1) on average compared to lexicographic order.
    
    def johnson_trotter(n):
        # Start with [1, 2, ..., n]
        perms = [[i for i in range(1, n + 1)]]
        directions = [1] * n # 1 for up, -1 for down
        
        while True:
            current = perms[-1]
            # Find the mobile element
            # An element is mobile if it is larger than its neighbor in the direction it is facing
            mobile_index = -1
            mobile_val = 0
            for i in range(n):
                val = current[i]
                dir = directions[i]
                if i + dir >= 0 and i + dir < n:
                    if val > current[i + dir]:
                        mobile_index = i
                        mobile_val = val
                        break
            
            if mobile_index == -1:
                break # No mobile element, done
            
            # Swap
            i = mobile_index
            j = i + directions[i]
            current[i], current[j] = current[j], current[i]
            
            # Reverse directions of all elements larger than the swapped one
            for k in range(n):
                if current[k] > mobile_val:
                    directions[k] = -directions[k]
            
            perms.append(current[:])
            
        return perms

    perms = johnson_trotter(n)
    
    # Stitch permutations
    current_word = list(perms[0])
    
    for i in range(1, len(perms)):
        next_perm = perms[i]
        # Find max overlap
        max_overlap = 0
        for k in range(n - 1, 0, -1):
            if tuple(current_word[-k:]) == tuple(next_perm[:k]):
                max_overlap = k
                break
        
        current_word.extend(next_perm[max_overlap:])
        
    return "".join(map(str, current_word))
