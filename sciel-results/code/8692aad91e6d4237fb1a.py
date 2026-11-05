import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # Precompute prime factors for all numbers 1..511
    # factors[i] will be a list of prime factors of i
    factors = [[] for _ in range(512)]
    for i in range(2, 512):
        if not factors[i]:
            for j in range(i, 512, i):
                if not factors[j]:
                    factors[j].append(i)
                else:
                    factors[j].append(i)
    
    # Helper to check if x is coprime to all elements in a list L
    def is_coprime_to_list(x, L):
        for a in L:
            if gcd(x, a) != 1:
                return False
        return True

    # Helper to calculate score
    def calc_score(A):
        return sum(1.0 / (512 - a) for a in A)

    # Initial Greedy Construction
    # We prioritize large numbers
    A = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in A):
            A.append(a)
    
    current_score = calc_score(A)
    
    # Local Search
    # Moves:
    # 1. Merge: Replace p, q in A with c = p*q if c < 512 and c is coprime to A \ {p, q}
    # 2. Swap: Replace a in A with b not in A if b is coprime to A \ {a} and score improves
    # 3. Add: Add b not in A if b is coprime to A and score improves (score always improves if added)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Merges (2-for-1)
        # Iterate over all pairs in A
        n = len(A)
        for i in range(n):
            if time.time() > deadline: break
            for j in range(i + 1, n):
                p = A[i]
                q = A[j]
                c = p * q
                if c >= 512:
                    continue
                
                # Check if c is coprime to all other elements in A
                # We need to check against A excluding p and q
                valid = True
                for k in range(n):
                    if k == i or k == j:
                        if gcd(c, A[k]) != 1:
                            valid = False
                            break
                
                if valid:
                    # Calculate score change
                    old_val = 1.0 / (512 - p) + 1.0 / (512 - q)
                    new_val = 1.0 / (512 - c)
                    
                    if new_val > old_val:
                        # Perform merge
                        # Remove p, q and add c
                        # We need to be careful with indices if we modify list in place
                        # Simpler: create new list
                        new_A = []
                        for k in range(n):
                            if k != i and k != j:
                                new_A.append(A[k])
                        new_A.append(c)
                        
                        # Update A and score
                        A = new_A
                        current_score += (new_val - old_val)
                        improved = True
                        n = len(A)
                        # Break out of inner loop to restart pair iteration with new A
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # 2. Try Adds
        # Check if any number not in A can be added
        # To be efficient, we can check numbers in descending order
        in_A = set(A)
        for b in range(511, 0, -1):
            if time.time() > deadline: break
            if b in in_A:
                continue
            
            if is_coprime_to_list(b, A):
                # Add b
                A.append(b)
                in_A.add(b)
                current_score += 1.0 / (512 - b)
                improved = True
                # Restart add loop to see if more can be added
                break
        
        if improved:
            continue
        
        # 3. Try Swaps (1-for-1)
        # Replace a in A with b not in A
        # This is trickier. We iterate a in A, and for each a, try to find a better b.
        # To limit search, we can look for b > a? Or just any b with higher value.
        # Higher value means b is closer to 512.
        # So for a given a, we look for b > a such that b is coprime to A \ {a}.
        
        n = len(A)
        for i in range(n):
            if time.time() > deadline: break
            a = A[i]
            
            # Try to find a b that replaces a
            # We want 1/(512-b) > 1/(512-a) => 512-b < 512-a => b > a
            # So we only need to check b > a
            # Also b must not be in A
            # And b must be coprime to A \ {a}
            
            # Optimization: Precompute A without a? Or just check on the fly.
            # Since N is small, on the fly is fine.
            
            found_swap = False
            # Iterate b from 511 down to a+1
            for b in range(511, a, -1):
                if b in in_A:
                    continue
                
                # Check if b is coprime to all a' in A where a' != a
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    if gcd(b, A[j]) != 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    old_val = 1.0 / (512 - a)
                    new_val = 1.0 / (512 - b)
                    
                    # Since b > a, new_val > old_val is guaranteed
                    A[i] = b
                    in_A.remove(a)
                    in_A.add(b)
                    current_score += (new_val - old_val)
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                break # Restart swap loop with new A

    return A
